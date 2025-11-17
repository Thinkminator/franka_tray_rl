import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# Load our XML model
model = mujoco.MjModel.from_xml_path("assets/panda_tray/world.xml")
data = mujoco.MjData(model)

# Define start and goal poses (replace with your actual values)
start_tray_pos = np.array([0.189, 0.186, 0.848])
goal_tray_pos = np.array([0.151, 0.614, 0.958])

# Find body IDs for the markers
start_marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "start_marker_body")
goal_marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_body")

if start_marker_body_id == -1 or goal_marker_body_id == -1:
    raise ValueError("Start or goal marker body not found in model")

# Get mocap IDs for these bodies
start_mocap_id = model.body_mocapid[start_marker_body_id]
goal_mocap_id = model.body_mocapid[goal_marker_body_id]

# Set mocap positions to start and goal positions
data.mocap_pos[start_mocap_id] = start_tray_pos
data.mocap_pos[goal_mocap_id] = goal_tray_pos

# Forward the simulation to update positions
mujoco.mj_forward(model, data)

# Use the same gains from franka_ros_controllers config
Kp = np.array([1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0])
Ki = np.zeros(7)
Kd = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

# Number of joints
n_joints = 7

# Desired initial joint positions (qpos0)
qpos0 = np.array([0.1, -1.24, 0.29, -2.3, 0.12, 2.68, -0.63], dtype=float)

# Convert 20 degrees to radians
rotation_angle = np.deg2rad(20.0)  # 20 degrees = 0.349 radians

# Trajectory timing parameters (expose these globally so loop can use them)
time_per_joint_move = 3.0
time_per_hold = 1.0
time_all_move = 4.0
time_all_hold = 2.0

phase1_duration = n_joints * (time_per_joint_move + time_per_hold)
phase2_duration = n_joints * (time_per_joint_move + time_per_hold)
phase3_duration = time_all_move + time_all_hold
phase4_duration = time_all_move + time_all_hold
total_cycle_duration = phase1_duration + phase2_duration + phase3_duration + phase4_duration

# Trajectory generator
def desired_trajectory(t, initial_positions):
    cycle_time = t % total_cycle_duration
    desired_positions = initial_positions.copy()

    # Phase 1: individual forward
    if cycle_time < phase1_duration:
        phase_time = cycle_time
        for joint_idx in range(n_joints):
            joint_start_time = joint_idx * (time_per_joint_move + time_per_hold)
            joint_end_time = joint_start_time + time_per_joint_move
            if joint_start_time <= phase_time < joint_end_time:
                progress = (phase_time - joint_start_time) / time_per_joint_move
                desired_positions[joint_idx] = initial_positions[joint_idx] + rotation_angle * progress
            elif phase_time >= joint_end_time:
                desired_positions[joint_idx] = initial_positions[joint_idx] + rotation_angle

    # Phase 2: individual backward
    elif cycle_time < phase1_duration + phase2_duration:
        phase_time = cycle_time - phase1_duration
        desired_positions = initial_positions + rotation_angle
        for joint_idx in range(n_joints):
            joint_start_time = joint_idx * (time_per_joint_move + time_per_hold)
            joint_end_time = joint_start_time + time_per_joint_move
            if joint_start_time <= phase_time < joint_end_time:
                progress = (phase_time - joint_start_time) / time_per_joint_move
                desired_positions[joint_idx] = (initial_positions[joint_idx] + rotation_angle) - rotation_angle * progress
            elif phase_time >= joint_end_time:
                desired_positions[joint_idx] = initial_positions[joint_idx]

    # Phase 3: all forward
    elif cycle_time < phase1_duration + phase2_duration + phase3_duration:
        phase_time = cycle_time - phase1_duration - phase2_duration
        if phase_time < time_all_move:
            progress = phase_time / time_all_move
            desired_positions = initial_positions + rotation_angle * progress
        else:
            desired_positions = initial_positions + rotation_angle

    # Phase 4: all backward
    else:
        phase_time = cycle_time - phase1_duration - phase2_duration - phase3_duration
        if phase_time < time_all_move:
            progress = phase_time / time_all_move
            desired_positions = (initial_positions + rotation_angle) - rotation_angle * progress
        else:
            desired_positions = initial_positions.copy()

    return desired_positions

# PID control per-joint (uses joint-specific gains)
def pid_control(target, current, prev_error, integral, dt, joint_idx):
    error = target - current
    integral += error * dt
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    control_signal = Kp[joint_idx] * error + Ki[joint_idx] * integral + Kd[joint_idx] * derivative
    return float(control_signal), float(error), float(integral)

# Get joint DOF indices and actuator ids
joint_names = [f"panda_joint{i+1}" for i in range(n_joints)]
joint_dof_ids = []
actuator_ids = []

for name in joint_names:
    dofadr = model.joint(name).dofadr
    # extract scalar dof index
    if hasattr(dofadr, "__len__"):
        dof_id = int(dofadr[0])
    else:
        dof_id = int(dofadr)
    joint_dof_ids.append(dof_id)
    actuator_ids.append(int(model.actuator(name).id))

print("Joint DOF IDs:", joint_dof_ids)
print("Actuator IDs:", actuator_ids)

# Set initial qpos to qpos0 using DOF indices, then forward the model
for i, dof_id in enumerate(joint_dof_ids):
    data.qpos[dof_id] = float(qpos0[i])
mujoco.mj_forward(model, data)

# Read back initial positions and bias (scalars)
initial_positions = np.zeros(n_joints)
for i, dof_id in enumerate(joint_dof_ids):
    pos = data.qpos[dof_id]
    initial_positions[i] = float(pos[0]) if hasattr(pos, "__len__") else float(pos)

prev_torque = np.zeros(n_joints)
for i, dof_id in enumerate(joint_dof_ids):
    bias = data.qfrc_bias[dof_id]
    prev_torque[i] = float(bias[0]) if hasattr(bias, "__len__") else float(bias)

print("Initial positions:", initial_positions)
print("Initial torque (bias):", prev_torque)

# State tracking for movement-completion printing
prev_phase = None            # 1,2,3,4
prev_moving_joint = None     # index of joint moving in phase1/2 or None
prev_moving_all = False      # True while in the "moving" part of phase3/4

# Main loop / viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    prev_error = np.zeros(n_joints)
    integral = np.zeros(n_joints)
    kDeltaTauMax = 1.0

    # Logging
    timestamps = []
    target_trajectories = [[] for _ in range(n_joints)]
    actual_positions = [[] for _ in range(n_joints)]
    control_signals = [[] for _ in range(n_joints)]
    torques_applied = [[] for _ in range(n_joints)]

    step_count = 0

    print("Starting trajectory...")

    while viewer.is_running():
        current_time = time.time() - start_time
        dt = model.opt.timestep

        # current positions (scalars)
        current_positions = np.zeros(n_joints)
        for i, dof_id in enumerate(joint_dof_ids):
            pos = data.qpos[dof_id]
            current_positions[i] = float(pos[0]) if hasattr(pos, "__len__") else float(pos)

        # Determine which phase and whether a joint (or all) is moving now
        cycle_time = current_time % total_cycle_duration

        # Default moving flags
        moving_joint = None
        moving_all = False
        phase = None

        if cycle_time < phase1_duration:
            phase = 1
            phase_time = cycle_time
            # find joint currently moving (if any)
            for joint_idx in range(n_joints):
                joint_start_time = joint_idx * (time_per_joint_move + time_per_hold)
                joint_end_time = joint_start_time + time_per_joint_move
                if joint_start_time <= phase_time < joint_end_time:
                    moving_joint = joint_idx
                    break
        elif cycle_time < phase1_duration + phase2_duration:
            phase = 2
            phase_time = cycle_time - phase1_duration
            for joint_idx in range(n_joints):
                joint_start_time = joint_idx * (time_per_joint_move + time_per_hold)
                joint_end_time = joint_start_time + time_per_joint_move
                if joint_start_time <= phase_time < joint_end_time:
                    moving_joint = joint_idx
                    break
        elif cycle_time < phase1_duration + phase2_duration + phase3_duration:
            phase = 3
            phase_time = cycle_time - phase1_duration - phase2_duration
            moving_all = (phase_time < time_all_move)
        else:
            phase = 4
            phase_time = cycle_time - phase1_duration - phase2_duration - phase3_duration
            moving_all = (phase_time < time_all_move)

        # Compute target positions for logging/control
        target_positions = desired_trajectory(current_time, initial_positions)

        # PID
        control_signals_vec = np.zeros(n_joints)
        for i in range(n_joints):
            cs, pe, integ = pid_control(float(target_positions[i]), float(current_positions[i]),
                                        float(prev_error[i]), float(integral[i]), dt, i)
            control_signals_vec[i] = cs
            prev_error[i] = pe
            integral[i] = integ

        # coriolis / gravity bias
        coriolis = np.zeros(n_joints)
        for i, dof_id in enumerate(joint_dof_ids):
            b = data.qfrc_bias[dof_id]
            coriolis[i] = float(b[0]) if hasattr(b, "__len__") else float(b)

        # torque command (PD + bias) with rate limiting
        tau_d_calculated = coriolis + control_signals_vec
        delta_tau = tau_d_calculated - prev_torque
        delta_tau = np.clip(delta_tau, -kDeltaTauMax, kDeltaTauMax)
        torque_command = prev_torque + delta_tau
        prev_torque = torque_command.copy()

        # apply torques
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = float(torque_command[i])

        # logging
        timestamps.append(float(current_time))
        for i in range(n_joints):
            target_trajectories[i].append(float(target_positions[i]))
            actual_positions[i].append(float(current_positions[i]))
            control_signals[i].append(float(control_signals_vec[i]))
            torques_applied[i].append(float(torque_command[i]))

        # --- detect movement completions and print desired vs actual ---
        # Individual joint moves (phase 1 & 2): print when a previously-moving joint finishes
        if phase in (1, 2):
            if prev_moving_joint is not None and prev_moving_joint != moving_joint:
                j = prev_moving_joint
                desired = target_trajectories[j][-1]  # the latest desired value for joint j
                actual = actual_positions[j][-1]
                err = desired - actual
                print(f"[{current_time:.3f}s] Finished movement of joint {j+1} (phase {prev_phase}): "
                      f"desired={desired:.4f} rad, actual={actual:.4f} rad, error={err:.4f} rad")
        # All-joint moves (phase 3 & 4): print when the moving_all window finishes
        else:
            if prev_moving_all and not moving_all and prev_phase in (3, 4):
                # movement of all joints just finished
                print(f"[{current_time:.3f}s] Finished all-joints movement (phase {prev_phase}):")
                for j in range(n_joints):
                    desired = target_trajectories[j][-1]
                    actual = actual_positions[j][-1]
                    err = desired - actual
                    print(f"   joint {j+1}: desired={desired:.4f}, actual={actual:.4f}, error={err:.4f}")

        # update trackers
        prev_phase = phase
        prev_moving_joint = moving_joint
        prev_moving_all = moving_all

        # step
        mujoco.mj_step(model, data)
        viewer.sync()

        step_count += 1
        time.sleep(max(0, dt - (time.time() - current_time)))

    # Plot results after viewer closed (optional)
    if len(timestamps) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        for i in range(n_joints):
            axes[0].plot(timestamps, target_trajectories[i], '--', label=f'Joint {i+1} Target', alpha=0.7)
            axes[0].plot(timestamps, actual_positions[i], '-', label=f'Joint {i+1} Actual', alpha=0.7)
        axes[0].set_ylabel('Joint Position (rad)')
        axes[0].set_title('Joint Position Tracking - All Joints')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)

        for i in range(n_joints):
            axes[1].plot(timestamps, control_signals[i], label=f'Joint {i+1} Control Signal')
        axes[1].set_ylabel('Control Signal')
        axes[1].set_title('PID Control Signals (All Joints)')
        axes[1].legend()
        axes[1].grid(True)

        for i in range(n_joints):
            axes[2].plot(timestamps, torques_applied[i], label=f'Joint {i+1} Torque')
        axes[2].set_ylabel('Torque (N⋅m)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Applied Torques (All Joints)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()