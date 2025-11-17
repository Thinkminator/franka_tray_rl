import os
import yaml
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from typing import Optional, Dict, Any, Tuple


class TrayPoseEnv(gym.Env):
    """
    Environment for tray-cylinder manipulation with a Panda robot using MuJoCo torque control.
    Actions are normalized joint velocity commands (7D), scaled to respect torque rate limits.
    The environment uses PD + feedforward torque control with torque-rate and torque limits.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 model_path: str = "assets/panda_tray/world.xml",
                 obs_noise_std_pos: Optional[float] = None,
                 obs_noise_std_vel: Optional[float] = None,
                 cylinder_noise_std_pos: Optional[float] = None,
                 cylinder_noise_std_vel: Optional[float] = None,
                 use_jacobian_tray_obs: Optional[bool] = None,
                 config_path: str = "envs/traypose/config.yaml"):
        super().__init__()

        # Load YAML config if available
        cfg = {}
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            print(f"Loaded config from {config_path}")
        else:
            print(f"Config file {config_path} not found or not loaded.")

        def get(path: str, default):
            cur = cfg
            try:
                for k in path.split("."):
                    cur = cur[k]
                return default if cur is None else cur
            except Exception:
                return default

        # Load MuJoCo model and data
        self.model_path = model_path or get("model_path", "assets/panda_tray/world.xml")
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Control parameters
        self.substeps = int(get("control.substeps", 10))
        self.sim_dt = float(self.model.opt.timestep)          # one mj_step duration
        self.control_dt = float(self.sim_dt * self.substeps)
        self.max_speed = float(get("control.max_speed", 0.3))  # rad/s for each joint

        # Debug prints
        self.debug_prints = bool(get("debug.debug_prints", False))
        self.debug_print_interval = int(get("debug.debug_interval_steps", 100))

        # Action space: normalized joint velocity commands [-1,1] for 7 joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation noise parameters
        self.obs_noise_std_pos = float(obs_noise_std_pos if obs_noise_std_pos is not None else get("observation.noise_std_pos", 0.0))
        self.obs_noise_std_vel = float(obs_noise_std_vel if obs_noise_std_vel is not None else get("observation.noise_std_vel", 0.0))
        self.cylinder_noise_std_pos = float(cylinder_noise_std_pos if cylinder_noise_std_pos is not None else get("observation.cylinder_noise_std_pos", 0.0))
        self.cylinder_noise_std_vel = float(cylinder_noise_std_vel if cylinder_noise_std_vel is not None else get("observation.cylinder_noise_std_vel", 0.0))
        self.use_jacobian_tray_obs = bool(use_jacobian_tray_obs if use_jacobian_tray_obs is not None else get("observation.use_jacobian_tray_obs", False))

        self.np_random = np.random.RandomState()

        # Observation space dimension (36)
        obs_dim = 36
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Start and goal configurations
        self.start_tray_pos = np.array(get("start.tray_pos", [0.189, 0.186, 0.848]), dtype=np.float64)
        self.start_tray_rpy = np.array(get("start.tray_rpy", [0.0, 0.0, -2.883]), dtype=np.float64)
        self.start_joint_positions = np.array(get("start.joints", [0.1, -1.24, 0.29, -2.3, 0.12, 2.68, -0.63]), dtype=np.float64)

        self.goal_tray_pos = np.array(get("goal.tray_pos", [0.151, 0.614, 0.958]), dtype=np.float64)
        self.goal_tray_rpy = np.array(get("goal.tray_rpy", [0.0, 0.0, -1.918]), dtype=np.float64)
        self.goal_pos_tolerance = float(get("goal.pos_tolerance", 0.03))
        self.goal_yaw_tolerance = float(get("goal.yaw_tolerance_deg", 5.0)) * np.pi / 180.0
        self._base_goal_yaw_tolerance = float(self.goal_yaw_tolerance)
        self._relaxed_goal_yaw_tolerance = float(get("goal.relaxed_yaw_tolerance_deg", 180.0)) * np.pi / 180.0
        self.success_hold_H = int(get("goal.success_hold_steps", 5))
        self.max_steps = int(get("goal.max_steps", 50000))

        # Cylinder start position
        self.start_cylinder = np.array(get("start.cylinder", [0.189, 0.186, 0.885]), dtype=np.float64)

        # Internal counters
        self.t = 0
        self.goal_hold_counter = 0

        # Joint and actuator mappings
        self.arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.arm_jnt_ids = []
        self.arm_qposadr = []
        self.arm_dofadr = []
        self.arm_act_ids = []

        for jname in self.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"Joint {jname} not found in model")
            self.arm_jnt_ids.append(jid)
            self.arm_qposadr.append(int(self.model.jnt_qposadr[jid]))
            self.arm_dofadr.append(int(self.model.jnt_dofadr[jid]))

            # Find actuator driving this joint
            act_id = None
            for a in range(self.model.nu):
                if self.model.actuator_trnid[a, 0] == jid:
                    act_id = a
                    break
            if act_id is None:
                raise ValueError(f"No actuator found for joint {jname}")
            self.arm_act_ids.append(act_id)

        # Cylinder joint info
        self.cylinder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
        if self.cylinder_joint == -1:
            raise ValueError("cylinder_free joint not found in MuJoCo model")
        self.cylinder_qposadr = int(self.model.jnt_qposadr[self.cylinder_joint])
        self.cylinder_dofadr = int(self.model.jnt_dofadr[self.cylinder_joint])
        self.cylinder_type = int(self.model.jnt_type[self.cylinder_joint])

        # Tray body id
        self.tray_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_base")
        if self.tray_body_id == -1:
            raise ValueError("Body 'tray_base' not found in MuJoCo model")

        # Controller gains and limits
        self.Kp = np.array(get("control.Kp", [1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0]), dtype=np.float64)
        self.Ki = np.array(get("control.Ki", [0.0]*7), dtype=np.float64)
        self.Kd = np.array(get("control.Kd", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]), dtype=np.float64)
        self.tau_limits = np.array(get("control.tau_limits", [87, 87, 87, 87, 12, 12, 12]), dtype=np.float64)
        self.delta_tau_max = float(get("control.delta_tau_max", 1.0))  # interpreted as Nm/s
        # Per-physics-step delta (used during each mj_step call)
        self.delta_tau_per_physics_step = self.delta_tau_max
        self.stabilization_steps = int(get("control.stabilization_steps", 50))


        # Joint limits
        self.joint_limits_low = np.array(get("limits.joint_low",
                                            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
                                         dtype=np.float64)
        self.joint_limits_high = np.array(get("limits.joint_high",
                                             [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
                                          dtype=np.float64)

        # Penalties and rewards
        self.penalty_base = float(get("penalties.base_step", -0.1))
        self.penalty_idle = float(get("penalties.idle_action", -0.4))
        self.penalty_drop = float(get("penalties.drop", -10.0))
        self.penalty_topple = float(get("penalties.topple", -5.0))
        self.slide_penalty_min = float(get("penalties.slide_penalty_min", -0.1))
        self.slide_penalty_max = float(get("penalties.slide_penalty_max", -0.5))
        self.slant_penalty_min = float(get("penalties.slant_penalty_min", -0.1))
        self.slant_penalty_max = float(get("penalties.slant_penalty_max", -0.5))
        self.progress_k = float(get("progress.progress_k", 5.0))
        self.min_delta = float(get("progress.min_delta", 5e-3))
        self.progress_max = float(get("progress.progress_max", 0.5))
        self.progress_min = float(get("progress.progress_min", 0.1))
        self.success_alpha = float(get("success_reward.alpha", 50.0))
        self.success_maxbonus = float(get("success_reward.max_bonus", 50.0))
        self.stay_reward = float(get("success_reward.stay_reward", 3.0))

        # Zones and thresholds
        self.slant_angle_min = float(get("slant_angle.slant_angle_min", 2))
        self.slant_angle_max = float(get("slant_angle.slant_angle_max", 45))
        self.rim_x_half = float(get("rim_zone.x_half", 0.05))
        self.rim_y_half = float(get("rim_zone.y_half", 0.09))
        self.slide_threshold = float(get("rim_zone.slide_threshold", 0.01))
        self.drop_center_margin = float(get("drop_check.center_to_center_margin", 0.09))

        # Curriculum training parameters
        self.num_phases = int(get("curriculum.num_phases", 5))
        self.current_phase = int(get("curriculum.current_phase", 1))
        self.phase_method = str(get("curriculum.phase_method", "linear"))
        self.phase_exp_rate = float(get("curriculum.phase_exp_rate", 3.0))
        self.consecutive_successes = 0

        # Desired joint target and torque history
        self.q_des = self.start_joint_positions.copy()
        self._prev_tau_applied = np.zeros(7, dtype=np.float64)

        # PID state (previous error and integral)
        self._prev_error = np.zeros(7, dtype=np.float64)
        self._integral = np.zeros(7, dtype=np.float64)

        # Velocity estimation helpers
        self.prev_tray_pos = None
        self.prev_tray_rpy = None
        self._prev_cyl_pos = None
        self.prev_joint_pos_noisy = None
        self.prev_joint_vel_noisy = None
        self.prev_cyl_angle = None

        # Marker body IDs
        self.start_marker_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "start_marker_body")
        self.goal_marker_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_body")
        if self.start_marker_body_id == -1 or self.goal_marker_body_id == -1:
            raise ValueError("Start or goal marker body not found in model")
        
        # Compute phase boundaries using flip exponential interpolation
        self.phase_boundaries = []
        for i in range(self.num_phases + 1):
            phase_pos = self.interpolate(
                i, 0, self.num_phases, 
                0.0, 1.0, 
                method=self.phase_method, exp_rate=self.phase_exp_rate
            )
            phase_point = self.start_tray_pos + phase_pos * (self.goal_tray_pos - self.start_tray_pos)
            self.phase_boundaries.append(phase_point)
        
        # Compute required consecutive successes per phase using flip exponential
        self.phase_success_thresholds = []
        max_success_threshold = 50  # Maximum consecutive successes needed
        for i in range(self.num_phases):
            threshold = int(self.interpolate(
                i, 0, self.num_phases - 1,
                1, max_success_threshold,
                method='flip_exponential', exp_rate=5.0
            ))
            self.phase_success_thresholds.append(threshold)
        
        # Set initial goal based on current phase
        self.current_goal_tray_pos = self.phase_boundaries[self.current_phase].copy()
        self.current_goal_tray_rpy = self.start_tray_rpy + (
            (self.current_phase / self.num_phases) * (self.goal_tray_rpy - self.start_tray_rpy)
        )

        # Defaults for curriculum restore
        self._defaults = dict(
            success_hold_H=self.success_hold_H,
            goal_tray_pos=self.goal_tray_pos.copy(),
            goal_tray_rpy=self.goal_tray_rpy.copy(),
            progress_k=self.progress_k,
            progress_min=self.progress_min,
            progress_max=self.progress_max,
            slant_angle_min=self.slant_angle_min,
            slant_penalty_min=self.slant_penalty_min,
            slant_penalty_max=self.slant_penalty_max,
            penalty_idle=self.penalty_idle,
        )

        # Logging
        print(f"\n{'='*60}")
        print(f"TrayPoseEnv initialized with TORQUE-BASED CONTROL")
        print(f"  Model: {self.model_path}")
        print(f"  Action space: 7D normalized joint velocity commands")
        print(f"  Max joint speed: {self.max_speed} rad/s")
        print(f"  Torque rate limit (delta_tau_max): {self.delta_tau_max} Nm/s")
        print(f"  Sticky q_des behavior: enabled (zero action holds last target)")
        print(f"  Observation noise: pos_std={self.obs_noise_std_pos:.4f} rad, vel_std={self.obs_noise_std_vel:.4f} rad/s")
        print(f"  Tray observation mode: {'Jacobian FK' if self.use_jacobian_tray_obs else 'Direct MuJoCo'}")
        print(f"  Max episode steps: {self.max_steps}")
        print(f"  Curriculum: {self.num_phases} phases")
        print("Actuator -> Joint mapping:")
        for i in range(7):
            try:
                act_name_start = self.model.name_actuatoradr[self.arm_act_ids[i]]
                act_name = self.model.names[act_name_start:].split(b'\x00')[0].decode('utf-8')
            except Exception:
                act_name = f"act_{self.arm_act_ids[i]}"
            print(f"  [{i}] actuator={self.arm_act_ids[i]:2d} -> joint={self.arm_jnt_ids[i]:2d} {self.arm_joint_names[i]:15s}")
        print(f"{'='*60}\n")

    # Utility functions
    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def interpolate(value, in_min, in_max, out_min, out_max, method='linear', exp_rate=5.0):
        if in_max == in_min:
            x = 1.0 if value >= in_max else 0.0
        else:
            x = (value - in_min) / (in_max - in_min)
            x = np.clip(x, 0.0, 1.0)

        if method == 'linear':
            y = x
        elif method == 'exponential':
            k = max(1e-8, float(exp_rate))
            y = (1.0 - np.exp(-k * x)) / (1.0 - np.exp(-k))
        elif method == 'flip_exponential':
            k = max(1e-8, float(exp_rate))
            y = 1.0 - (1.0 - np.exp(-k * (1.0 - x))) / (1.0 - np.exp(-k))
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return out_min + y * (out_max - out_min)

    def _update_markers(self):
        start_mocap_id = self.model.body_mocapid[self.start_marker_body_id]
        goal_mocap_id = self.model.body_mocapid[self.goal_marker_body_id]
        self.data.mocap_pos[start_mocap_id] = self.start_tray_pos
        self.data.mocap_pos[goal_mocap_id] = self.current_goal_tray_pos
        mujoco.mj_forward(self.model, self.data)
    
    def _set_arm_qpos(self, qpos: np.ndarray):
        """Set the arm joint positions in the simulation state."""
        for i, q in enumerate(qpos):
            qpos_addr = self.arm_qposadr[i]
            self.data.qpos[qpos_addr] = float(q)

    def _get_arm_qpos(self, noisy=False):
        q = np.array([self.data.qpos[addr] for addr in self.arm_qposadr], dtype=np.float64)
        if noisy and self.obs_noise_std_pos > 0:
            q += self.np_random.normal(0, self.obs_noise_std_pos, size=q.shape)
        return q

    def _get_arm_qvel(self, noisy=False):
        qd = np.array([self.data.qvel[addr] for addr in self.arm_dofadr], dtype=np.float64)
        if noisy and self.obs_noise_std_vel > 0:
            qd += self.np_random.normal(0, self.obs_noise_std_vel, size=qd.shape)
        return qd
    
    def _compute_tray_fk_jacobian(self, joint_pos, joint_vel):
        temp_data = mujoco.MjData(self.model)
        for i, addr in enumerate(self.arm_qposadr):
            temp_data.qpos[addr] = joint_pos[i]
        for i, addr in enumerate(self.arm_dofadr):
            temp_data.qvel[addr] = joint_vel[i]
        mujoco.mj_forward(self.model, temp_data)

        pos = temp_data.xpos[self.tray_body_id].copy()
        quat_wxyz = temp_data.xquat[self.tray_body_id].copy()  # [w,x,y,z]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rpy = R.from_quat(quat_xyzw).as_euler('xyz')

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, temp_data, jacp, jacr, self.tray_body_id)
        jacp_arm = jacp[:, self.arm_dofadr]
        jacr_arm = jacr[:, self.arm_dofadr]

        linear_vel = jacp_arm @ joint_vel
        angular_vel = jacr_arm @ joint_vel
        return pos, rpy, linear_vel, angular_vel

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.t = 0
        self.goal_hold_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint positions
        self._set_arm_qpos(self.start_joint_positions)
        for dof in self.arm_dofadr:
            self.data.qvel[dof] = 0.0

        # Initialize cylinder
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+7] = np.array(
                [*self.start_cylinder, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.data.qvel[self.cylinder_dofadr:self.cylinder_dofadr+6] = 0.0

        # Update markers and do a forward pass
        self._update_markers()
        mujoco.mj_forward(self.model, self.data)

        # Initialize q_des and prev torque from MuJoCo bias (prevents droop)
        self.q_des = self.start_joint_positions.copy()
        self._prev_tau_applied = np.array(self.data.qfrc_bias[:7], dtype=np.float64)

        # Reset PID integrators & previous error
        self._prev_error.fill(0.0)
        self._integral.fill(0.0)

        # --- Stabilization loop: apply zero-action (i.e. keep q_des) torques for a number of physics steps ---
        # This lets floating objects and small transient dynamics settle.
        for _ in range(int(self.stabilization_steps)):
            # Write current prev torque (bias) to actuators
            for i, act_id in enumerate(self.arm_act_ids):
                self.data.ctrl[act_id] = float(self._prev_tau_applied[i])
            # Step the physics one mj_step at a time
            mujoco.mj_step(self.model, self.data)

        # After stabilization steps, run forward and refresh the bias/prev-torque so it matches settled state
        mujoco.mj_forward(self.model, self.data)
        self._prev_tau_applied = np.array(self.data.qfrc_bias[:7], dtype=np.float64)

        # Reset tracking vars using the settled state
        self.prev_tray_pos = self.data.xpos[self.tray_body_id].copy()
        self.prev_tray_rpy = R.from_quat([
            self.data.xquat[self.tray_body_id][1],
            self.data.xquat[self.tray_body_id][2],
            self.data.xquat[self.tray_body_id][3],
            self.data.xquat[self.tray_body_id][0]
        ]).as_euler('xyz').copy()

        self.prev_goal_dist = float(np.linalg.norm(self.start_tray_pos - self.current_goal_tray_pos))

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.last_action = np.array(action)
        self.t += 1

        # 1. Denormalize action to joint velocities (already clipped earlier)
        qd_cmd = action * self.max_speed  # rad/s

        # 2. Integrate q_des using physics timestep (same as action_changing.py)
        dt = float(self.model.opt.timestep)
        self.q_des += qd_cmd * dt
        self.q_des = np.clip(self.q_des, self.joint_limits_low, self.joint_limits_high)

        # 3. Read current joint positions
        q_current = self._get_arm_qpos(noisy=False)

        # 4. PID on position error (P + I + D) same as action_changing.py
        error = self.q_des - q_current
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else np.zeros_like(error)
        control_signal = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = error.copy()

        # 5. Coriolis / gravity bias from MuJoCo (no extra factor)
        coriolis = np.array(self.data.qfrc_bias[:7], dtype=np.float64)

        # 6. Total desired torque before limits
        tau_d_calculated = coriolis + control_signal

        # 7. Clip to actuator torque limits (if any)
        tau_d_calculated = np.clip(tau_d_calculated, -self.tau_limits, self.tau_limits)

        # 8. Rate-limited torque command (apply rate limit and then execute one mj_step, same as action_changing)
        delta_tau = tau_d_calculated - self._prev_tau_applied
        delta_tau = np.clip(delta_tau, -self.delta_tau_per_physics_step, self.delta_tau_per_physics_step)
        torque_command = self._prev_tau_applied + delta_tau
        self._prev_tau_applied = torque_command.copy()

        # 9. Apply torques to actuators
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = float(torque_command[i])

        # 10. Step the simulation once (same as action_changing.py)
        mujoco.mj_step(self.model, self.data)

        # 11. Read updated joint positions after stepping
        q_current = self._get_arm_qpos(noisy=False)

        # Inside step() method, after computing q_current and q_des
        if self.debug_prints and (self.t % self.debug_print_interval == 0):
            print(f"[DEBUG] Step {self.t}: q_desired = {np.round(self.q_des, 4)}")
            print(f"[DEBUG] Step {self.t}: q_actual  = {np.round(q_current, 4)}")
            print(f"[DEBUG] Step {self.t}: q_error = {np.round(self.q_des - q_current, 4)}")

        # 7. Update tray pose for reward/termination
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()
        self.tray_rpy = R.from_quat([tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]).as_euler('xyz')
        self.tray_yaw = self.tray_rpy[2]

        # 8. Get observation and calculate reward
        obs = self._get_obs()
        cyl_pos_w, _ = self._get_cylinder_world_pos_vel()
        reward, terminated, truncated, info = self._calculate_reward_and_done(cyl_pos_w)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _calculate_reward_and_done(self, cyl_pos_w):
        """
        Calculate reward, termination conditions, and info dictionary.
        
        Args:
            cyl_pos_w: Cylinder position in world coordinates
            
        Returns:
            reward (float): The reward for this step
            terminated (bool): Whether the episode should terminate
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information about the step
        """
        # Initialize return values
        reward = 0.0
        terminated = False
        truncated = False
        is_success = False
        drop_terminated = False
        topple_terminated = False
        cyl_angle = 0.0

        # Goal proximity
        yaw_err = self._wrap_angle(self.tray_yaw - self.current_goal_tray_rpy[2])
        pos_err = np.linalg.norm(self.tray_pos - self.current_goal_tray_pos)
        at_goal_now = (pos_err <= self.goal_pos_tolerance) and (abs(yaw_err) <= self.goal_yaw_tolerance)

        # Maintain goal hold counter
        if at_goal_now:
            self.goal_hold_counter += 1
        else:
            self.goal_hold_counter = 0

        # Base time penalty every step
        reward = self.penalty_base

        action_mag = float(np.linalg.norm(self.last_action))
        if not at_goal_now:
            if action_mag <= 1e-4:
                reward += self.penalty_idle
        else:
            reward += self.stay_reward

        # --- Progress reward (distance-proportional, per-step) ---
        current_goal_dist = float(np.linalg.norm(self.tray_pos - self.current_goal_tray_pos))
        delta_goal = self.prev_goal_dist - current_goal_dist  # positive if closer

        if delta_goal > self.min_delta:
            progress_reward = np.clip(self.progress_k * delta_goal, self.progress_min, self.progress_max)
            reward += progress_reward

        # Update previous goal distance
        self.prev_goal_dist = current_goal_dist

        # Continuous sliding penalty based on distance of cylinder from center to rim
        rel_cyl_xy_world = cyl_pos_w[:2] - self.tray_pos[:2]
        distance = np.linalg.norm(rel_cyl_xy_world)

        rim_x_half = self.rim_x_half  
        rim_y_half = self.rim_y_half
        rim_diagonal = np.sqrt(rim_x_half**2 + rim_y_half**2)

        if distance <= rim_diagonal and distance >= self.slide_threshold:
            # Clamp distance to the interpolation range
            sliding_penalty = self.interpolate(
                    distance,
                    self.slide_threshold, rim_diagonal,
                    self.slide_penalty_min, self.slide_penalty_max,
                    method='exponential', exp_rate=5.0
                    )
            reward += sliding_penalty

        # Continuous slanting penalty based on angle of tray
        slant_angle_min_rad = np.deg2rad(self.slant_angle_min)
        slant_angle_max_rad = np.deg2rad(self.slant_angle_max)
        roll_abs = abs(self.tray_rpy[0])
        pitch_abs = abs(self.tray_rpy[1])
                
        # Interpolate roll penalty
        if roll_abs >= slant_angle_min_rad:
            roll_penalty = self.interpolate(
                roll_abs,
                slant_angle_min_rad, slant_angle_max_rad,
                self.slant_penalty_min, self.slant_penalty_max,
                method='flip_exponential', exp_rate=5.0
            )
            reward += roll_penalty

        # Interpolate pitch penalty
        if pitch_abs >= slant_angle_min_rad:
            pitch_penalty = self.interpolate(
                pitch_abs,
                slant_angle_min_rad, slant_angle_max_rad,
                self.slant_penalty_min, self.slant_penalty_max,
                method='flip_exponential', exp_rate=5.0
            )
            reward += pitch_penalty

        # Drop termination (large penalty)
        if cyl_pos_w[2] < (self.tray_pos[2] - self.drop_center_margin):
            reward += self.penalty_drop
            terminated = True
            drop_terminated = True

        # Topple termination (medium penalty)
        if not terminated and self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            cyl_quat = self.data.qpos[self.cylinder_qposadr+3:self.cylinder_qposadr+7]
            cyl_rot = R.from_quat([cyl_quat[1], cyl_quat[2], cyl_quat[3], cyl_quat[0]])
            z_axis = cyl_rot.apply([0, 0, 1])
            angle_from_upright = np.arccos(np.clip(z_axis[2], -1.0, 1.0))
            if angle_from_upright > slant_angle_max_rad:  # > 45 deg from upright
                reward += self.penalty_topple
                terminated = True
                topple_terminated = True
                cyl_angle = angle_from_upright

        # Success check: if held goal for H steps, grant final positive reward
        if not terminated and self.goal_hold_counter >= self.success_hold_H:
            cyl_offset = np.linalg.norm(rel_cyl_xy_world)
            final_bonus = max(0.0, self.success_maxbonus - self.success_alpha * cyl_offset)
            reward += final_bonus
            is_success = True
            terminated = True  # task finishes successfully

        # Time limit truncation
        if not terminated and self.t >= self.max_steps:
            truncated = True
        
        current_phase = self.current_phase
        consecutive_successes = self.consecutive_successes
        success_threshold = self.phase_success_thresholds[self.current_phase - 1] if self.current_phase <= self.num_phases else 0

        info = {
            'goal_hold_counter': self.goal_hold_counter,
            'at_goal': at_goal_now,
            'pos_err': pos_err if terminated else 0.0,
            'yaw_err': yaw_err if terminated else 0.0,
            'cylinder_offset': np.linalg.norm(rel_cyl_xy_world) if terminated else 0.0,
            'cylinder_angle': cyl_angle if topple_terminated else 0.0,
            'terminated_due_to_drop': drop_terminated,
            'terminated_due_to_topple': topple_terminated,
            'is_success': is_success,
            'truncated': truncated,
            'phase': current_phase,
            'consecutive_successes': consecutive_successes,
            'success_threshold': success_threshold
        }
        
        return reward, terminated, truncated, info

    def _get_cylinder_world_pos_vel(self):
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            pos = self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
            vel = self.data.qvel[self.cylinder_dofadr:self.cylinder_dofadr+3].copy()
        else:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
                pos = self.data.xpos[bid].copy()
                if hasattr(self.data, "xvelp"):
                    vel = self.data.xvelp[bid].copy()
                else:
                    if self._prev_cyl_pos is None:
                        vel = np.zeros(3)
                    else:
                        vel = (pos - self._prev_cyl_pos) / self.control_dt
                    self._prev_cyl_pos = pos.copy()
            except Exception:
                try:
                    pos = self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
                except Exception:
                    pos = np.zeros(3)
                vel = np.zeros(3)
        return pos, vel
    
    def _get_tray_pose_velocity(self, use_noisy_joints=False):
        if self.use_jacobian_tray_obs:
            joint_pos = self._get_arm_qpos(noisy=use_noisy_joints)
            joint_vel = self._get_arm_qvel(noisy=use_noisy_joints)
            return self._compute_tray_fk_jacobian(joint_pos, joint_vel)
        else:
            pos = self.data.xpos[self.tray_body_id].copy()
            quat_wxyz = self.data.xquat[self.tray_body_id].copy()
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rpy = R.from_quat(quat_xyzw).as_euler('xyz')

            if hasattr(self.data, "xvelp") and hasattr(self.data, "xvelr"):
                try:
                    linear_vel = self.data.xvelp[self.tray_body_id].copy()
                    angular_vel = self.data.xvelr[self.tray_body_id].copy()
                except Exception:
                    linear_vel = None
                    angular_vel = None
            else:
                linear_vel = None
                angular_vel = None

            if linear_vel is None or angular_vel is None:
                if self.prev_tray_pos is None:
                    linear_vel = np.zeros(3)
                    angular_vel = np.zeros(3)
                else:
                    linear_vel = (pos - self.prev_tray_pos) / self.control_dt

                    def angle_diff(a, b):
                        return (a - b + np.pi) % (2 * np.pi) - np.pi
                    delta_rpy = np.array([
                        angle_diff(rpy[0], self.prev_tray_rpy[0]),
                        angle_diff(rpy[1], self.prev_tray_rpy[1]),
                        angle_diff(rpy[2], self.prev_tray_rpy[2])
                    ])
                    angular_vel = delta_rpy / self.control_dt

                self.prev_tray_pos = pos.copy()
                self.prev_tray_rpy = rpy.copy()

            return pos, rpy, linear_vel, angular_vel

    def _get_obs(self):
        # Compose observation vector with noisy joint states, tray pose/velocities, goal pose, and cylinder info
        current_tray_pos, tray_rpy, tray_linear_velocity, tray_angular_velocity = self._get_tray_pose_velocity(use_noisy_joints=True)

        tray_quat_xyzw = R.from_euler('xyz', tray_rpy).as_quat()
        cyl_pos_w, cyl_vel_w = self._get_cylinder_world_pos_vel()
        R_tray_w = R.from_quat(tray_quat_xyzw).as_matrix()
        R_w_tray = R_tray_w.T

        rel_pos_w = cyl_pos_w - current_tray_pos
        rel_pos_tray = R_w_tray @ rel_pos_w
        rel_vel_w = cyl_vel_w - tray_linear_velocity
        rel_vel_tray = R_w_tray @ rel_vel_w

        cyl_xy_in_tray = rel_pos_tray[:2].copy()
        cyl_vxy_in_tray = rel_vel_tray[:2].copy()

        if self.cylinder_noise_std_pos > 0.0:
            cyl_xy_in_tray += self.np_random.normal(0.0, self.cylinder_noise_std_pos, size=2)
        if self.cylinder_noise_std_vel > 0.0:
            cyl_vxy_in_tray += self.np_random.normal(0.0, self.cylinder_noise_std_vel, size=2)

        cyl_angle = 0.0
        cyl_angle_rate = 0.0
        try:
            if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
                cyl_quat = self.data.qpos[self.cylinder_qposadr+3:self.cylinder_qposadr+7]
                cyl_rot = R.from_quat([cyl_quat[1], cyl_quat[2], cyl_quat[3], cyl_quat[0]])
                z_axis = cyl_rot.apply([0, 0, 1])
                angle_from_upright = np.arccos(np.clip(z_axis[2], -1.0, 1.0))
                cyl_angle = angle_from_upright
                if self.prev_cyl_angle is None:
                    cyl_angle_rate = 0.0
                else:
                    delta = (cyl_angle - self.prev_cyl_angle + np.pi) % (2 * np.pi) - np.pi
                    cyl_angle_rate = float(delta / self.control_dt)
        except Exception:
            cyl_angle = 0.0
            cyl_angle_rate = 0.0
        self.prev_cyl_angle = float(cyl_angle)

        joint_angles = self._get_arm_qpos(noisy=True)
        joint_velocities = self._get_arm_qvel(noisy=True)
        goal_pose = np.concatenate([self.current_goal_tray_pos, np.array([self.current_goal_tray_rpy[2]])])
        tilt = np.array([cyl_angle, cyl_angle_rate])

        return np.concatenate([
            joint_angles,            # 7
            joint_velocities,        # 7
            current_tray_pos,        # 3
            tray_rpy,                # 3
            tray_linear_velocity,    # 3
            tray_angular_velocity,   # 3
            goal_pose,               # 4
            cyl_xy_in_tray,          # 2
            cyl_vxy_in_tray,         # 2
            tilt                     # 2
        ]).astype(np.float32)

    def render(self, mode="human"):
        cylinder_state = self._get_cylinder_world_pos_vel()[0]
        print(f"[LOG] Step {self.t}: Tray position {self.tray_pos}, Cylinder position {cylinder_state}")

    def close(self):
        pass

    def update_curriculum(self, success: bool):
        """
        Update the curriculum based on episode success.
        
        Args:
            success (bool): Whether the episode was successful
        """
        if success:
            self.consecutive_successes += 1
            # Check if we should advance to the next phase
            if (self.current_phase <= self.num_phases and 
                self.consecutive_successes >= self.phase_success_thresholds[self.current_phase - 1]):
                self.advance_phase()
        else:
            # Reset consecutive successes on failure
            self.consecutive_successes = 0

    def advance_phase(self):
        """Advance to the next curriculum phase."""
        if self.current_phase < self.num_phases:
            # advance using set_phase so yaw tolerance and markers are updated consistently
            self.set_phase(self.current_phase + 1)
            self.consecutive_successes = 0  # Reset counter for new phase
            print(f"[LOG] Advanced to phase {self.current_phase}/{self.num_phases}")
        elif self.current_phase == self.num_phases:
            print("[LOG] Completed all curriculum phases!")

    def set_phase(self, phase: int):
        """Set curriculum phase and update goal + yaw tolerance.

        Phase numbering follows self.current_phase (1..num_phases).
        For phases 1..(num_phases-1) we use a relaxed yaw tolerance; for the last phase
        we restore the original tight yaw tolerance.
        """
        # clamp
        phase = int(phase)
        if phase < 1:
            phase = 1
        if phase > getattr(self, "num_phases", phase):
            phase = self.num_phases

        self.current_phase = phase
        # keep the older 'phase' attribute in sync if used elsewhere
        self.phase = phase

        # set yaw tolerance: relax for all but final phase
        if self.current_phase < self.num_phases:
            self.goal_yaw_tolerance = float(self._relaxed_goal_yaw_tolerance)
        else:
            self.goal_yaw_tolerance = float(self._base_goal_yaw_tolerance)

        # update goal position/orientation and markers
        self._update_phase_goal()

        print(f"[LOG] set_phase -> {self.current_phase}/{self.num_phases}, "
              f"goal_yaw_tolerance_deg={np.rad2deg(self.goal_yaw_tolerance):.1f}")

    def _update_phase_goal(self):
        """Update the goal position based on the current phase."""
        self.current_goal_tray_pos = self.phase_boundaries[self.current_phase].copy()
        # Linearly interpolate orientation
        progress = self.current_phase / self.num_phases
        self.current_goal_tray_rpy = self.start_tray_rpy + progress * (self.goal_tray_rpy - self.start_tray_rpy)
        
        # Update markers to show new goal
        self._update_markers()