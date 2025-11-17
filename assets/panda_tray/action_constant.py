#!/usr/bin/env python3
import sys
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- Settings ----------
Kp = np.array([1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0])
Ki = np.zeros(7)
Kd = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

n_joints = 7

# initial qpos0 (kept as before)
qpos0 = np.array([0.1, -1.24, 0.29, -2.3, 0.12, 2.68, -0.63], dtype=float)

# max angular speed for denormalization (rad/s)
max_speed = 0.3

# torque rate limiter (you can relax for testing)
kDeltaTauMax = 1.0

# simulation length guard (optional)
max_steps = 5000

# ---------- Helper functions ----------
def pid_control(target, current, prev_error, integral, dt, joint_idx):
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    control_signal = Kp[joint_idx] * error + Ki[joint_idx] * integral + Kd[joint_idx] * derivative
    return float(control_signal), float(error), float(integral)

def extract_scalar(x):
    # Handles mujoco scalar or 1-element arrays returned by some mj bindings
    if hasattr(x, "__len__"):
        return float(x[0])
    return float(x)

# ---------- Parse mode ----------
if len(sys.argv) < 2:
    print("Usage: python action.py [zero|seeded|random]")
    sys.exit(1)

mode = sys.argv[1].lower()
if mode not in ("zero", "seeded", "random"):
    print("Mode must be 'zero', 'seeded' or 'random'")
    sys.exit(1)

# ---------- Load model ----------
model = mujoco.MjModel.from_xml_path("assets/panda_tray/world.xml")
data = mujoco.MjData(model)

# Get joint names / DOF / actuator ids
joint_names = [f"panda_joint{i+1}" for i in range(n_joints)]
joint_dof_ids = []
actuator_ids = []
for name in joint_names:
    dofadr = model.joint(name).dofadr
    dof_id = int(dofadr[0]) if hasattr(dofadr, "__len__") else int(dofadr)
    joint_dof_ids.append(dof_id)
    actuator_ids.append(int(model.actuator(name).id))

print("Joint DOF IDs:", joint_dof_ids)
print("Actuator IDs:", actuator_ids)

# ---------- Initialize state ----------
# Set initial qpos
for i, dof_id in enumerate(joint_dof_ids):
    data.qpos[dof_id] = float(qpos0[i])
mujoco.mj_forward(model, data)

# Read initial positions as scalars
initial_positions = np.zeros(n_joints)
for i, dof_id in enumerate(joint_dof_ids):
    initial_positions[i] = extract_scalar(data.qpos[dof_id])

# initialize prev_torque to bias to prevent droop
prev_torque = np.zeros(n_joints)
for i, dof_id in enumerate(joint_dof_ids):
    prev_torque[i] = extract_scalar(data.qfrc_bias[dof_id])

print("Initial positions:", np.round(initial_positions, 4))
print("Initial torque (bias):", np.round(prev_torque, 4))

# ---------- Create action vector ----------
rng = np.random.default_rng(12345)  # deterministic RNG for seeded mode
if mode == "zero":
    action_norm = np.zeros(n_joints)
elif mode == "seeded":
    action_norm = rng.uniform(-1.0, 1.0, size=n_joints)
else:  # random
    action_norm = np.random.uniform(-1.0, 1.0, size=n_joints)

# show chosen normalized action and denormalized speed
vel_cmd = action_norm * max_speed  # rad/s
print(f"Mode: {mode}")
print("Normalized action:", np.round(action_norm, 4))
print("Denormalized joint speeds (rad/s):", np.round(vel_cmd, 6))

# q_des starts at initial positions
q_des = initial_positions.copy()

# Logging
timestamps = []
target_trajectories = [[] for _ in range(n_joints)]
actual_positions = [[] for _ in range(n_joints)]
control_signals = [[] for _ in range(n_joints)]
torques_applied = [[] for _ in range(n_joints)]

# ---------- Run simulation with viewer ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    prev_error = np.zeros(n_joints)
    integral = np.zeros(n_joints)

    step_count = 0
    print("Starting action-driven control loop. Close the viewer window to stop.")

    while viewer.is_running() and step_count < max_steps:
        current_time = time.time() - start_time
        dt = model.opt.timestep

        # integrate q_des using denormalized velocity * dt (fixed action)
        delta_q = vel_cmd * dt
        q_des += delta_q

        # read current positions (scalars)
        current_positions = np.zeros(n_joints)
        for i, dof_id in enumerate(joint_dof_ids):
            current_positions[i] = extract_scalar(data.qpos[dof_id])

        # PID to compute control signal
        control_signals_vec = np.zeros(n_joints)
        for i in range(n_joints):
            cs, pe, integ = pid_control(q_des[i], current_positions[i], prev_error[i], integral[i], dt, i)
            control_signals_vec[i] = cs
            prev_error[i] = pe
            integral[i] = integ

        # gravity / Coriolis bias
        coriolis = np.zeros(n_joints)
        for i, dof_id in enumerate(joint_dof_ids):
            coriolis[i] = extract_scalar(data.qfrc_bias[dof_id])

        # total desired torque (bias + PID)
        tau_d_calculated = coriolis + control_signals_vec

        # rate-limited torque command (same as before)
        delta_tau = tau_d_calculated - prev_torque
        delta_tau = np.clip(delta_tau, -kDeltaTauMax, kDeltaTauMax)
        torque_command = prev_torque + delta_tau
        prev_torque = torque_command.copy()

        # apply torques to actuators
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = float(torque_command[i])

        # logging
        timestamps.append(float(current_time))
        for i in range(n_joints):
            target_trajectories[i].append(float(q_des[i]))
            actual_positions[i].append(float(current_positions[i]))
            control_signals[i].append(float(control_signals_vec[i]))
            torques_applied[i].append(float(torque_command[i]))

        # occasional print to monitor progress
        if step_count % 500 == 0:
            print(f"[{current_time:.2f}s] step {step_count} q_des (first4): {np.round(q_des[:4],4)} q_act (first4): {np.round(current_positions[:4],4)}")

        # step and viewer sync
        mujoco.mj_step(model, data)
        viewer.sync()

        step_count += 1
        time.sleep(max(0, dt - (time.time() - current_time)))

    print("Control loop ended. step_count =", step_count)

# ---------- After viewer closed: print final desired vs actual and plot ----------
final_errors = []
print("\nFinal desired vs actual (rad):")
for i in range(n_joints):
    desired = target_trajectories[i][-1]
    actual = actual_positions[i][-1]
    err = desired - actual
    final_errors.append(abs(err))
    print(f" joint {i+1}: desired={desired:.4f}, actual={actual:.4f}, error={err:.4f}")

print(f"Mean absolute error: {np.mean(final_errors):.6f} rad")

# save plots to file (no blocking)
if len(timestamps) > 0:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for i in range(n_joints):
        axes[0].plot(timestamps, target_trajectories[i], '--', alpha=0.7, label=f'j{i+1} target' if i==0 else None)
        axes[0].plot(timestamps, actual_positions[i], '-', alpha=0.7, label=f'j{i+1} actual' if i==0 else None)
    axes[0].set_ylabel('Joint Position (rad)')
    axes[0].set_title('Target vs Actual (all joints)')
    axes[0].grid(True)

    for i in range(n_joints):
        axes[1].plot(timestamps, control_signals[i], label=f'j{i+1}' if i<6 else None)
    axes[1].set_ylabel('Control signal')
    axes[1].grid(True)

    for i in range(n_joints):
        axes[2].plot(timestamps, torques_applied[i], label=f'j{i+1}' if i<6 else None)
    axes[2].set_ylabel('Torque (N·m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)

    plt.tight_layout()
    out_name = f"action_run_{mode}.png"
    plt.savefig(out_name, dpi=200)
    print(f"Saved plot to {out_name}")

print("Done.")