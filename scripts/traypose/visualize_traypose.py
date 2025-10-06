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

def print_space_info(env):
    """Print detailed action and observation space information"""
    print(f"\n{'='*60}")
    print("ACTION SPACE")
    print(f"{'='*60}")
    print(f"Type: {type(env.action_space)}")
    print(f"Shape: {env.action_space.shape}")
    print(f"Low bounds: {env.action_space.low}")
    print(f"High bounds: {env.action_space.high}")
    print(f"Dtype: {env.action_space.dtype}")
    print(f"\nAction interpretation:")
    print(f"  - 7D vector (one per Panda joint)")
    print(f"  - Normalized range: [-1, 1]")
    print(f"  - Maps to joint position increments")
    print(f"  - Max increment per step: {env.max_joint_increment} rad")
    print(f"  - Max speed: {env.max_joint_increment/env.control_dt:.2f} rad/s")
    
    print(f"\n{'='*60}")
    print("OBSERVATION SPACE")
    print(f"{'='*60}")
    print(f"Type: {type(env.observation_space)}")
    print(f"Shape: {env.observation_space.shape}")
    print(f"Low bounds: {env.observation_space.low[:5]}... (showing first 5)")
    print(f"High bounds: {env.observation_space.high[:5]}... (showing first 5)")
    print(f"Dtype: {env.observation_space.dtype}")
    
    print(f"\nObservation breakdown (34 dimensions):")
    print(f"  [0:2]   Cylinder XY in tray frame (2D)")
    print(f"  [2:5]   Tray XYZ position (3D)")
    print(f"  [5:8]   Tray roll, pitch, yaw (3D)")
    print(f"  [8:11]  Tray linear velocity (3D)")
    print(f"  [11:14] Tray angular velocity (3D)")
    print(f"  [14:21] Joint angles (7D)")
    print(f"  [21:28] Joint velocities (7D)")
    print(f"  [28:32] Goal pose (XYZ + yaw, 4D)")
    print(f"  [32:34] Cylinder velocity XY in tray frame (2D)")
    print(f"{'='*60}\n")

def print_observation_details(obs, step=0):
    """Print detailed observation values"""
    print(f"\n--- Observation at step {step} ---")
    print(f"Cylinder XY (tray):  {obs[0:2]}")
    print(f"Tray position:       {obs[2:5]}")
    print(f"Tray RPY:            {obs[5:8]}")
    print(f"Tray lin vel:        {obs[8:11]}")
    print(f"Tray ang vel:        {obs[11:14]}")
    print(f"Joint angles:        {obs[14:21]}")
    print(f"Joint velocities:    {obs[21:28]}")
    print(f"Goal pose:           {obs[28:32]}")
    print(f"Cylinder vel XY:     {obs[32:34]}")

def run_zero_action_mode(env, viewer, episode, pause_seconds=3.0):
    """Mode 1: Zero action - arm stays at start pose"""
    print(f"\n=== Episode {episode+1} - ZERO ACTION MODE ===")
    
    # Reset env
    obs = env.reset()
    
    # Print initial observation
    print_observation_details(obs, step=0)
    
    print(f"\nHolding start pose with zero actions for {pause_seconds}s...")
    n_steps = int(pause_seconds / env.control_dt)
    
    action = np.zeros(7, dtype=np.float32)
    print(f"Action (zero): {action}")
    
    for step in range(n_steps):
        # Apply zero action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Print status every second
        if step % int(1.0 / env.control_dt) == 0:
            cylinder_pos = env._get_cylinder_xyz()
            tray_pos = env.tray_pos
            cyl_xy_tray = obs[0:2]
            cyl_vel_tray = obs[32:34]
            print(f"  t={step*env.control_dt:.1f}s | Reward: {reward:.3f}")
            print(f"    Tray pos: {tray_pos}")
            print(f"    Cylinder world: {cylinder_pos}")
            print(f"    Cylinder XY (tray frame): {cyl_xy_tray}")
            print(f"    Cylinder vel XY (tray frame): {cyl_vel_tray}")
            print(f"    Terminated: {terminated}, Truncated: {truncated}, HoldCounter: {info.get('goal_hold_counter')}")
        
        viewer.sync()
        time.sleep(env.control_dt)
        
        if done or not viewer.is_running():
            if done:
                reason = "success" if terminated and info.get("at_goal", False) else ("time_limit" if truncated else "terminated")
                print(f"Episode ended early: {reason}")
            break
    
    # Print final observation
    print_observation_details(obs, step=min(step+1, n_steps))
    print(f"Zero action mode completed after {min(step+1, n_steps)} steps")

def run_random_action_mode(env, viewer, episode, max_steps=500):
    """Mode 2: Random action - arm moves randomly within limits"""
    print(f"\n=== Episode {episode+1} - RANDOM ACTION MODE ===")
    
    # Reset env
    obs = env.reset()
    print_observation_details(obs, step=0)
    print(f"\nRunning random actions for up to {max_steps} steps...")

    step_count = 0
    
    while step_count < max_steps and viewer.is_running():
        # Sample random action in [-1, 1]
        action = np.random.uniform(-1.0, 1.0, size=7).astype(np.float32)
        
        # We keep moving even if terminated/truncated to observe free motion, but we log it.
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if step_count % 50 == 0:
            cylinder_pos = env._get_cylinder_xyz()
            tray_pos = env.tray_pos
            cyl_xy_tray = obs[0:2]
            cyl_vel_tray = obs[32:34]
            joint_pos = obs[14:21]          
            joint_vel = obs[21:28]
            print(f"\n  Step {step_count}")
            print(f"    Action: {action}")
            print(f"    Reward: {reward:.3f}")
            print(f"    Tray pos: {tray_pos}")
            print(f"    Cylinder world pos: {cylinder_pos}")
            print(f"    Cylinder XY (tray frame): {cyl_xy_tray}")
            print(f"    Cylinder vel XY (tray frame): {cyl_vel_tray}")
            print(f"    Joint pos: {joint_pos}")    
            print(f"    Joint vel: {joint_vel}")   
            print(f"    Terminated: {terminated}, Truncated: {truncated}, HoldCounter: {info.get('goal_hold_counter')}")
            print(f"    Obs shape: {obs.shape}, sample: {obs[:5]}...")
        
        viewer.sync()
        step_count += 1
        time.sleep(env.control_dt)
    
    # Final check and observation
    cylinder_pos = env._get_cylinder_xyz()
    print(f"\nRandom action mode finished after {step_count} steps")
    print_observation_details(obs, step=step_count)

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
    
    # Initialize environment with joint-space torque PD + gravity compensation
    env = TrayPoseEnv()
    
    # # Light noise (realistic sensor noise)
    # env = TrayPoseEnv(obs_noise_std_pos=0.001, obs_noise_std_vel=0.01)  # ~0.06° pos, 0.57°/s vel

    # # Moderate noise
    # env = TrayPoseEnv(obs_noise_std_pos=0.005, obs_noise_std_vel=0.05)  # ~0.29° pos, 2.86°/s vel

    # # Heavy noise (stress test)
    # env = TrayPoseEnv(obs_noise_std_pos=0.01, obs_noise_std_vel=0.1)    # ~0.57° pos, 5.73°/s vel

    # Jacobian FK with noisy joints → noisy tray obs
    env = TrayPoseEnv(obs_noise_std_pos=0.005, obs_noise_std_vel=0.05, use_jacobian_tray_obs=True)

    # Print space info
    print_space_info(env)

    print("\n🔧 Autotuning hold-pose gains...")
    env.autotune_hold_pose(hold_seconds=2.0, max_pos_error_rad=0.01, verbose=True)
    print("✓ Autotuning complete.\n")
        
    print(f"\nEnvironment settings:")
    print(f"  Control dt: {env.control_dt}s ({1/env.control_dt:.0f} Hz)")
    print(f"  Substeps: {env.substeps}")
    print(f"  Observation noise: pos_std={env.obs_noise_std_pos:.4f} rad, vel_std={env.obs_noise_std_vel:.4f} rad/s")
    
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