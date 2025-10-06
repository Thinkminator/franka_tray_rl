import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
import mujoco
import mujoco.viewer

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# Add project root to sys.path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now we can import from envs.traypose
from envs.traypose.traypose_env import TrayPoseEnv

def evaluate_model(model_path, num_episodes=3):
    # Create environment
    env = TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")
    
    # Load trained model
    model = PPO.load(model_path, env=env)

    all_rewards = []
    
    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            step_count = 0
            
            print(f"\nStarting Episode {ep+1}")
            
            while not done and viewer.is_running():
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take step in environment (Gymnasium API)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                
                # Update MuJoCo internal state
                mujoco.mj_forward(env.model, env.data)
                
                # Render in viewer
                viewer.sync()
                
                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}, Reward: {reward:.3f}, Terminated={terminated}, Truncated={truncated}")
                
                step_count += 1
                time.sleep(env.control_dt)  # Maintain real-time speed
            
            print(f"Episode {ep+1} finished after {step_count} steps with reward: {ep_reward:.3f}")
            all_rewards.append(ep_reward)

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.3f}")

if __name__ == "__main__":
    model_path = "training/checkpoints/best_model.zip"
    evaluate_model(model_path)