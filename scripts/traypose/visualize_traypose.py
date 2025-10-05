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

def check_cylinder_spawn(env):
    """Check if cylinder is properly spawned and visible"""
    cylinder_pos = env._get_cylinder_xyz()
    print(f"  Cylinder position: {cylinder_pos}")
    
    # Check if position is valid (not NaN or at origin)
    if np.any(np.isnan(cylinder_pos)):
        print("  ⚠️  WARNING: Cylinder position contains NaN!")
        return False
    elif np.allclose(cylinder_pos, [0, 0, 0]):
        print("  ⚠️  WARNING: Cylinder at origin (may not be spawned)!")
        return False
    else:
        print("  ✓ Cylinder spawned successfully")
        return True

def run_zero_action_mode(env, viewer, episode, pause_seconds=3.0):
    """Mode 1: Zero action - arm stays at start pose"""
    print(f"\n=== Episode {episode+1} - ZERO ACTION MODE ===")
    
    # Reset env
    obs = env.reset()
    
    # Check cylinder spawn
    check_cylinder_spawn(env)
    
    print(f"Holding start pose with zero actions for {pause_seconds}s...")
    n_steps = int(pause_seconds / env.control_dt)
    
    for step in range(n_steps):
        # Apply zero action
        obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
        
        # Print status every second
        if step % int(1.0 / env.control_dt) == 0:
            cylinder_pos = env._get_cylinder_xyz()
            tray_pos = env.tray_pos
            print(f"  t={step*env.control_dt:.1f}s | Tray: {tray_pos} | Cylinder: {cylinder_pos}")
        
        viewer.sync()
        time.sleep(env.control_dt)
        
        if done or not viewer.is_running():
            break
    
    print(f"Zero action mode completed after {n_steps} steps")

def run_random_action_mode(env, viewer, episode, max_steps=500):
    """Mode 2: Random action - arm moves randomly within limits"""
    print(f"\n=== Episode {episode+1} - RANDOM ACTION MODE ===")
    
    # Reset env
    obs = env.reset()
    
    # Check cylinder spawn
    check_cylinder_spawn(env)
    
    print(f"Running random actions for up to {max_steps} steps...")
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps and viewer.is_running():
        # Sample random action in [-1, 1]
        action = np.random.uniform(-1.0, 1.0, size=7).astype(np.float32)
        
        obs, reward, done, info = env.step(action)
        
        # Print status every 50 steps
        if step_count % 50 == 0:
            cylinder_pos = env._get_cylinder_xyz()
            tray_pos = env.tray_pos
            print(f"  Step {step_count} | Reward: {reward:.3f} | Tray: {tray_pos[:2]} | Cyl: {cylinder_pos[:2]}")
        
        viewer.sync()
        step_count += 1
        time.sleep(env.control_dt)
    
    # Final check
    cylinder_pos = env._get_cylinder_xyz()
    print(f"Random action mode finished after {step_count} steps")
    print(f"  Final cylinder position: {cylinder_pos}")
    if cylinder_pos[2] < 0.1:
        print("  ⚠️  Cylinder dropped!")

def main():
    # Parse command line argument for mode
    mode = "zero"  # default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ["zero", "random"]:
        print("Usage: python visualize_traypose.py [zero|random]")
        print("  zero   - Zero action mode (arm stays at start pose)")
        print("  random - Random action mode (arm moves randomly)")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running in {mode.upper()} mode")
    print(f"{'='*60}")
    
    # Initialize environment with direct position control (no PID)
    env = TrayPoseEnv(
        model_path="assets/panda_tray/panda_tray_cylinder.xml",
        max_joint_increment=0.01  # 0.01 rad/step ≈ 0.5 rad/s at 50Hz
    )
    
    print(f"\nEnvironment settings:")
    print(f"  Control dt: {env.control_dt}s ({1/env.control_dt:.0f} Hz)")
    print(f"  Max joint increment: {env.max_joint_increment} rad/step")
    print(f"  Max joint speed: {env.max_joint_increment/env.control_dt:.2f} rad/s")
    print(f"  Substeps: {env.substeps}")
    
    # Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        num_episodes = 3
        
        for episode in range(num_episodes):
            if mode == "zero":
                run_zero_action_mode(env, viewer, episode, pause_seconds=3.0)
            else:  # random
                run_random_action_mode(env, viewer, episode, max_steps=500)
            
            # Pause between episodes
            if episode < num_episodes - 1:
                print("\nPausing 1s before next episode...")
                time.sleep(1.0)
    
    env.close()
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()