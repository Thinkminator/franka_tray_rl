### TrayPose Environment

#### Overview
TrayPoseEnv simulates a Franka Panda manipulating a tray with a free cylinder in MuJoCo. The agent issues 7D normalized joint increment actions; a joint-space PD controller with gravity compensation converts these into joint torques. The environment supports noisy observations and two ways of generating tray pose/velocity for observations: directly from MuJoCo or via Jacobian-based forward kinematics from joint states.

- Control: joint-space torque PD + gravity compensation
- Robot: Franka Panda with a tray fixed at end effector
- Actuation: 7 revolute joints (panda_joint1..7)
- Object: a cylinder (free joint)
- Step time: control_dt with substeps internal physics updates
- Configuration: via config.yaml, with constructor args overriding YAML

#### Action Space
- Type: Box(low=-1, high=1, shape=(7,))
- Semantics: per-step joint increments (normalized)
  - dq_cmd = action * max_joint_increment [rad]
  - q_des = clip(q + dq_cmd, joint_limits)
- Important tunables:
  - control.control_dt (s): env step time
  - control.substeps (int): physics steps per env step
  - control.max_joint_increment (rad/step): action scale
  - limits.joint_low / limits.joint_high (rad): joint limits

Example:
```python
# Default scaling
env = TrayPoseEnv()

# Tighter per-step increments
env = TrayPoseEnv(max_joint_increment=0.02)
```

#### Observation Space (34D)
Order and meaning:
1) Robot state
- [0:7] joint angles q (7D) [rad]
- [7:14] joint velocities dq (7D) [rad/s]

2) Tray pose and dynamics (either MuJoCo direct or Jacobian FK)
- [14:17] tray position [x, y, z] (world)
- [17:20] tray orientation [roll, pitch, yaw] (rad)
- [20:23] tray linear velocity [vx, vy, vz] (world)
- [23:26] tray angular velocity [wx, wy, wz] (body/world convention from MuJoCo/Jacobian)

3) Goal
- [26:30] [goal_x, goal_y, goal_z, goal_yaw] (rad)

4) Cylinder state in tray frame
- [30:32] cylinder XY in tray frame (m)
- [32:34] cylinder XY velocities in tray frame (m/s)

Noise:
- observation.noise_std_pos adds Gaussian noise to q
- observation.noise_std_vel adds Gaussian noise to dq
- use_jacobian_tray_obs controls whether tray pose/vel in obs are derived via Jacobian from the (potentially noisy) joints or read directly from MuJoCo
- cylinder_noise_std_pos adds Gaussian noise to the cylinder relative position
- cylinder_noise_std_vel adds Gaussian noise to the cylinder relative velocity 

Examples:
```python
# Clean observations from MuJoCo (default)
env = TrayPoseEnv()

# Noisy joint observations (muJoCo tray obs)
env = TrayPoseEnv(obs_noise_std_pos=0.005, obs_noise_std_vel=0.05)

# Noisy joint observations with Jacobian FK tray obs
env = TrayPoseEnv(obs_noise_std_pos=0.005, obs_noise_std_vel=0.05, use_jacobian_tray_obs=True)

# Noisy cylinder observation and joint observation with Jacobian FK tray obs
env = TrayPoseEnv(obs_noise_std_pos=0.005, obs_noise_std_vel=0.05, cylinder_noise_std_pos=0.005, cylinder_noise_std_vel=0.01, use_jacobian_tray_obs=True)
```

Precedence:
- Constructor args override YAML values; YAML overrides hardcoded defaults.

#### Control and Dynamics
- Internal controller computes torque:
  - tau = Kq · (q_des − q) − Dq · dq + qfrc_bias
  - qfrc_bias provides gravity, Coriolis, and centrifugal compensation per MuJoCo
- Gains and torque limits:
  - pd_gains.Kq (7D), pd_gains.Dq (7D)
  - control.tau_limits (7D) applied element-wise
- q_des “sticky”: if action is zero, desired target holds, producing a hold-position behavior

Autotuning (optional):
- env.autotune_hold_pose(...) adjusts Kq and Dq scalars for proximal/distal groups to hold pose under gravity with zero action.

#### Reward Structure
Base and penalties (all configurable):
- Step penalty: penalties.base_step (e.g., −0.1)
- Idle penalty if small action: penalties.idle_action (e.g., −0.4)
- Sliding penalty: if cylinder XY slides more than slide_threshold, a negative penalty that is proportional to the distance will be given
- Slanting penalty: if cylinder XY slants more than slant_angle_min, a negative penalty that is proportional to the angle will be given
- Progress reward: if the current distance between tray and goal pose is smaller than previous one, a reward that is proportional to the substraction will be given


Termination penalties:
- Drop: if cylinder z < tray_z − drop_margin → add penalties.drop and terminate
- Topple: if cylinder axis tilts > 45° from upright → add penalties.topple and terminate

Success bonus:
- If at goal and held for goal.success_hold_steps:
  - Compute cyl_offset = ||cylinder_xy_world − tray_xy_world||
  - final_bonus = max(0, success_reward.max_bonus − success_reward.alpha · cyl_offset)
  - Add bonus and terminate

Goal conditions:
- Within goal.pos_tolerance (m) and goal.yaw_tolerance_deg (converted to rad)
- Must be satisfied for consecutive success_hold_steps

Time limit:
- Truncation at goal.max_steps if not terminated

Info keys:
- goal_hold_counter, at_goal, pos_err, yaw_err, cylinder_offset

#### Termination and Truncation Summary
- terminated = True if:
  - Drop condition met
  - Topple condition met
  - Success hold satisfied (after awarding final bonus)
- truncated = True if:
  - Step count reaches max_steps without termination

#### Modes in visualize_traypose.py
- Zero action mode:
  - Applies zero actions for a fixed duration to verify hold pose and observe stability
- Random action mode:
  - Applies random actions; logs state every 50 steps
- Seeded action mode:
  - Applies random actions from a fixed seed; logs state every 50 steps 

Selecting mode:
```python
python scripts/traypose/visualize_traypose.py zero
```
```python
python scripts/traypose/visualize_traypose.py random
```
```python
python scripts/traypose/visualize_traypose.py seeded
```

Viewer:
- Uses mujoco.viewer.launch_passive to render in real time
- The script prints action/observation space details and sample observations

#### Configuration (config.yaml)
Key blocks:
- model_path: XML path
- control: control_dt, substeps, max_joint_increment, tau_limits
- observation: noise_std_pos, noise_std_vel, use_jacobian_tray_obs, cylinder_noise_std_pos, cylinder_noise_std_vel
- start: tray_pos, tray_rpy, joints, cylinder
- goal: tray_pos, tray_rpy, pos_tolerance, yaw_tolerance_deg, success_hold_steps, max_steps
- pd_gains: Kq, Dq
- limits: joint_low, joint_high
- penalties: base_step, idle_action, rim, drop, topple, lide_penalty_min, slide_penalty_max, slant_penalty_min, slant_penalty_max, progress_k, progress_max
- progress: min_delta
- rim_zone: x/y bounds in tray frame, slide_threshold
- slant_angle: slant_angle_min/max
- drop_check: center_to_center_margin
- success_reward: alpha, max_bonus

Constructor overrides:
```python
# YAML is loaded by default from config.yaml; args below override YAML values
env = TrayPoseEnv(
    obs_noise_std_pos=0.005,
    obs_noise_std_vel=0.05,
    use_jacobian_tray_obs=True,
    cylinder_noise_std_pos: 0.005,
    cylinder_noise_std_vel: 0.01
)
```

#### Quick Reference of Defaults (via YAML)
- control_dt = 0.02 s, substeps = 10
- max_joint_increment = 0.04 rad/step
- tau_limits = [87, 87, 87, 87, 12, 12, 12]
- noise_std_pos = 0.0, noise_std_vel = 0.0
- use_jacobian_tray_obs = false
- cylinder_noise_std_pos: 0.0, cylinder_noise_std_vel: 0.0
- success bonus: max_bonus = 5.0, alpha = 50.0
- penalties: base −0.1, idle −0.4, rim −0.5, drop −10, topple −5
- drop margin: 0.09 m
- goal: pos_tolerance 0.03 m, yaw_tolerance 5°, hold 5 steps, max_steps 150
