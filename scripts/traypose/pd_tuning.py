import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import os
import sys

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# Load the model
model = mujoco.MjModel.from_xml_path('assets/panda_tray/panda_tray_cylinder.xml')
data = mujoco.MjData(model)

# Get joint indices
joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
               'panda_joint5', 'panda_joint6', 'panda_joint7']
n_joints = len(joint_names)

# Joint limits
joint_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
joint_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# PD Controller class with per-joint gains
class PDController:
    def __init__(self, kp, kd):
        """
        kp, kd: arrays of length 7 (one gain per joint)
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
    
    def compute_torque(self, q_target, q_current, qd_current):
        """Compute PD control torque per joint"""
        error = q_target - q_current
        torque = self.kp * error - self.kd * qd_current
        return torque

def simulate_with_pd(kp, kd, target_angles, duration=3.0):
    """
    Simulate the robot with given PD gains
    Returns: mean tracking error, max tracking error, settling time
    """
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position to home configuration
    data.qpos[:n_joints] = np.zeros(n_joints)
    mujoco.mj_forward(model, data)
    
    # Create controller
    controller = PDController(kp, kd)
    
    # Simulation parameters
    dt = model.opt.timestep
    n_steps = int(duration / dt)
    
    # Storage for analysis
    errors = []
    joint_errors = []
    positions = []
    velocities = []
    times = []
    
    for step in range(n_steps):
        # Get current state
        q_current = data.qpos[:n_joints].copy()
        qd_current = data.qvel[:n_joints].copy()
        
        # Compute control torque
        torque = controller.compute_torque(target_angles, q_current, qd_current)
        
        # Clip torques to actuator limits
        torque_limits = np.array([87, 87, 87, 87, 12, 12, 12])
        torque = np.clip(torque, -torque_limits, torque_limits)
        
        # Apply control
        data.ctrl[:n_joints] = torque
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data
        error = np.linalg.norm(target_angles - q_current)
        errors.append(error)
        joint_errors.append(np.abs(target_angles - q_current))
        positions.append(q_current.copy())
        velocities.append(qd_current.copy())
        times.append(step * dt)
    
    errors = np.array(errors)
    joint_errors = np.array(joint_errors)
    positions = np.array(positions)
    velocities = np.array(velocities)
    times = np.array(times)
    
    # Calculate metrics
    mean_error = np.mean(errors[-500:])  # Last 1 second
    max_error = np.max(errors)
    mean_joint_errors = np.mean(joint_errors[-500:], axis=0)
    
    # Settling time (when error stays below 5% of initial error)
    threshold = 0.05 * errors[0] if errors[0] > 0 else 0.01
    settled_idx = np.where(errors < threshold)[0]
    settling_time = times[settled_idx[0]] if len(settled_idx) > 0 else duration
    
    # Check cylinder is free (should have moved due to gravity)
    cylinder_z = data.qpos[-7]  # z position of cylinder freejoint
    cylinder_moved = abs(cylinder_z - 0.655) > 0.01
    
    return mean_error, max_error, settling_time, errors, times, positions, cylinder_moved, mean_joint_errors

# Test different PD gain combinations
print("Testing PD controller gains (per-joint tuning)...")
print("=" * 80)

# Define test target angles (moderate movement for initial tuning)
target_angles = np.array([0.5, -0.3, 0.4, -1.5, 0.2, 1.0, 0.3])

# Grid search for PD gains - now per joint
# Joints 1-4 are larger/stronger (87 Nm limit), joints 5-7 are smaller (12 Nm limit)
kp_values_large = [100, 200, 400, 800]  # For joints 1-4
kp_values_small = [50, 100, 200, 400]   # For joints 5-7
kd_values_large = [10, 20, 40, 80]      # For joints 1-4
kd_values_small = [5, 10, 20, 40]       # For joints 5-7

# Start with best uniform gains from previous run
best_uniform_kp = 800
best_uniform_kd = 10

# Initialize with uniform gains
best_kp = np.ones(n_joints) * best_uniform_kp
best_kd = np.ones(n_joints) * best_uniform_kd

print(f"Target angles: {target_angles}")
print(f"\nStarting with uniform gains: Kp={best_uniform_kp}, Kd={best_uniform_kd}")

# Baseline performance
mean_err, max_err, settle_time, errors, times, positions, cyl_moved, joint_errs = \
    simulate_with_pd(best_kp, best_kd, target_angles, duration=3.0)
best_cost = mean_err * 100 + max_err * 10 + settle_time

print(f"Baseline cost: {best_cost:.2f}")
print(f"Per-joint errors: {joint_errs}")
print()

# Tune each joint individually
print("Tuning each joint individually...")
print("-" * 80)

for joint_idx in range(n_joints):
    print(f"\nTuning Joint {joint_idx + 1} ({joint_names[joint_idx]})...")
    
    # Select appropriate gain ranges
    if joint_idx < 4:  # Large joints
        kp_range = kp_values_large
        kd_range = kd_values_large
    else:  # Small joints
        kp_range = kp_values_small
        kd_range = kd_values_small
    
    joint_best_cost = best_cost
    joint_best_kp = best_kp[joint_idx]
    joint_best_kd = best_kd[joint_idx]
    
    for kp_test in kp_range:
        for kd_test in kd_range:
            # Test this gain combination for this joint
            test_kp = best_kp.copy()
            test_kd = best_kd.copy()
            test_kp[joint_idx] = kp_test
            test_kd[joint_idx] = kd_test
            
            mean_err, max_err, settle_time, _, _, _, cyl_moved, _ = \
                simulate_with_pd(test_kp, test_kd, target_angles, duration=3.0)
            
            cost = mean_err * 100 + max_err * 10 + settle_time
            
            if cost < joint_best_cost:
                joint_best_cost = cost
                joint_best_kp = kp_test
                joint_best_kd = kd_test
                print(f"  Joint {joint_idx+1}: Kp={kp_test:4d}, Kd={kd_test:3d} | Cost: {cost:.2f} ✓")
    
    # Update best gains for this joint
    best_kp[joint_idx] = joint_best_kp
    best_kd[joint_idx] = joint_best_kd
    best_cost = joint_best_cost
    
    print(f"  → Best for Joint {joint_idx+1}: Kp={joint_best_kp}, Kd={joint_best_kd}")

# Final evaluation with optimized gains
print("\n" + "=" * 80)
print("FINAL OPTIMIZED PD GAINS (PER-JOINT):")
print("=" * 80)

for i in range(n_joints):
    print(f"Joint {i+1} ({joint_names[i]:15s}): Kp = {best_kp[i]:6.1f}, Kd = {best_kd[i]:5.1f}")

print("=" * 80)

# ============================================================================
# TEST ACROSS MULTIPLE TARGET POSITIONS
# ============================================================================
print("\n" + "=" * 80)
print("TESTING ACROSS MULTIPLE TARGET POSITIONS")
print("=" * 80)

# Generate test positions: low, mid-low, center, mid-high, high for each joint
n_test_positions = 5
test_targets = []

for i in range(n_test_positions):
    alpha = i / (n_test_positions - 1)  # 0, 0.25, 0.5, 0.75, 1.0
    target = joint_low + alpha * (joint_high - joint_low)
    test_targets.append(target)

test_targets = np.array(test_targets)

print(f"\nTesting {n_test_positions} different target configurations...")
print("Target positions:")
for i, target in enumerate(test_targets):
    print(f"  Config {i+1}: {target}")

# Store results for all test positions
all_results = []

for config_idx, target in enumerate(test_targets):
    print(f"\nTesting configuration {config_idx + 1}/{n_test_positions}...")
    
    mean_err, max_err, settle_time, errors, times, positions, cyl_moved, joint_errs = \
        simulate_with_pd(best_kp, best_kd, target, duration=3.0)
    
    all_results.append({
        'target': target,
        'positions': positions,
        'times': times,
        'errors': errors,
        'joint_errors': joint_errs,
        'mean_error': mean_err,
        'max_error': max_err,
        'settling_time': settle_time
    })
    
    print(f"  Mean error: {mean_err:.4f} rad, Max error: {max_err:.4f} rad, Settling: {settle_time:.2f}s")

# ============================================================================
# PLOT INDIVIDUAL JOINT TRACKING
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING INDIVIDUAL JOINT PLOTS")
print("=" * 80)

# Create a figure with 7 subplots (one per joint)
fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.flatten()

colors = plt.cm.viridis(np.linspace(0, 1, n_test_positions))

for joint_idx in range(n_joints):
    ax = axes[joint_idx]
    
    for config_idx, result in enumerate(all_results):
        target_val = result['target'][joint_idx]
        positions = result['positions'][:, joint_idx]
        times = result['times']
        
        # Plot actual position
        ax.plot(times, positions, color=colors[config_idx], linewidth=1.5,
                label=f'Target={target_val:.2f}')
        
        # Plot target as horizontal line
        ax.axhline(y=target_val, color=colors[config_idx], linestyle='--', 
                   linewidth=1, alpha=0.6)
    
    # Add joint limits as shaded regions
    ax.axhspan(joint_low[joint_idx], joint_high[joint_idx], alpha=0.1, color='green')
    ax.axhline(y=joint_low[joint_idx], color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=joint_high[joint_idx], color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Position (rad)', fontsize=10)
    ax.set_title(f'{joint_names[joint_idx]} (Kp={best_kp[joint_idx]:.0f}, Kd={best_kd[joint_idx]:.0f})', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

# Remove the extra subplot (we have 7 joints, 8 subplot positions)
fig.delaxes(axes[7])

plt.tight_layout()
plt.savefig('individual_joint_tracking.png', dpi=150, bbox_inches='tight')
print("Saved: individual_joint_tracking.png")

# ============================================================================
# SUMMARY PLOT
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall tracking error for each configuration
axes2[0, 0].bar(range(1, n_test_positions + 1), 
                [r['mean_error'] for r in all_results],
                color='steelblue', alpha=0.7)
axes2[0, 0].set_xlabel('Configuration', fontsize=12)
axes2[0, 0].set_ylabel('Mean Tracking Error (rad)', fontsize=12)
axes2[0, 0].set_title('Mean Error Across Configurations', fontsize=14)
axes2[0, 0].set_xticks(range(1, n_test_positions + 1))
axes2[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Settling time for each configuration
axes2[0, 1].bar(range(1, n_test_positions + 1), 
                [r['settling_time'] for r in all_results],
                color='coral', alpha=0.7)
axes2[0, 1].set_xlabel('Configuration', fontsize=12)
axes2[0, 1].set_ylabel('Settling Time (s)', fontsize=12)
axes2[0, 1].set_title('Settling Time Across Configurations', fontsize=14)
axes2[0, 1].set_xticks(range(1, n_test_positions + 1))
axes2[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Per-joint mean errors (heatmap)
joint_error_matrix = np.array([r['joint_errors'] for r in all_results])
im = axes2[1, 0].imshow(joint_error_matrix.T, aspect='auto', cmap='YlOrRd', origin='lower')
axes2[1, 0].set_xlabel('Configuration', fontsize=12)
axes2[1, 0].set_ylabel('Joint', fontsize=12)
axes2[1, 0].set_title('Per-Joint Mean Errors', fontsize=14)
axes2[1, 0].set_xticks(range(n_test_positions))
axes2[1, 0].set_xticklabels(range(1, n_test_positions + 1))
axes2[1, 0].set_yticks(range(n_joints))
axes2[1, 0].set_yticklabels([f'J{i+1}' for i in range(n_joints)])
plt.colorbar(im, ax=axes2[1, 0], label='Error (rad)')

# Plot 4: PD gains bar chart
x = np.arange(n_joints)
width = 0.35
axes2[1, 1].bar(x - width/2, best_kp, width, label='Kp', color='steelblue', alpha=0.7)
axes2[1, 1].bar(x + width/2, best_kd, width, label='Kd', color='coral', alpha=0.7)
axes2[1, 1].set_xlabel('Joint', fontsize=12)
axes2[1, 1].set_ylabel('Gain Value', fontsize=12)
axes2[1, 1].set_title('Optimized PD Gains', fontsize=14)
axes2[1, 1].set_xticks(x)
axes2[1, 1].set_xticklabels([f'J{i+1}' for i in range(n_joints)])
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pd_tuning_summary.png', dpi=150, bbox_inches='tight')
print("Saved: pd_tuning_summary.png")

plt.show()

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print("\nRecommended per-joint PD gains:")
print("kp = np.array([" + ", ".join([f"{k:.1f}" for k in best_kp]) + "])")
print("kd = np.array([" + ", ".join([f"{k:.1f}" for k in best_kd]) + "])")
print("\nAverage performance across all configurations:")
print(f"  Mean error: {np.mean([r['mean_error'] for r in all_results]):.4f} rad")
print(f"  Max error:  {np.max([r['max_error'] for r in all_results]):.4f} rad")
print(f"  Avg settling time: {np.mean([r['settling_time'] for r in all_results]):.2f} s")
print("=" * 80)