import gym
import numpy as np
import mujoco
from gym import spaces
from scipy.spatial.transform import Rotation as R


class TrayPoseEnv(gym.Env):
    """
    Tray-cylinder manipulation with Panda using torque control.
    Actions are normalized joint increments (7D).
    Internal joint-space PD controller with gravity compensation computes joint torques.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path="assets/panda_tray/panda_tray_cylinder.xml",
                 max_joint_increment=0.04,
                 obs_noise_std_pos=0.0,
                 obs_noise_std_vel=0.0,
                 use_jacobian_tray_obs=False):
        """
        Args:
            model_path: Path to MuJoCo XML model
            max_joint_increment: Maximum joint increment per step (radians)
            obs_noise_std_pos: Standard deviation of Gaussian noise for joint positions (radians)
            obs_noise_std_vel: Standard deviation of Gaussian noise for joint velocities (rad/s)
            use_jacobian_tray_obs: If True, compute tray pose/vel from noisy joint states via Jacobian FK.
                                   If False, use direct MuJoCo body pose (default behavior).
        """
        super().__init__()

        # Load MuJoCo model
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control settings
        self.control_dt = 0.02     # env step time
        self.substeps = 10         # physics substeps per env step
        self.sim_dt = self.model.opt.timestep
        
        # Action: normalized joint increments (7 joints)
        self.max_joint_increment = float(max_joint_increment)  # radians per step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation noise settings
        self.obs_noise_std_pos = float(obs_noise_std_pos)
        self.obs_noise_std_vel = float(obs_noise_std_vel)
        self.use_jacobian_tray_obs = bool(use_jacobian_tray_obs)

        # Observation: 34 dims (added cylinder planar velocity 2D)
        obs_dim = 34
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # START configuration
        self.start_tray_pos = np.array([0.788, 0.108, 0.621], dtype=np.float64)
        self.start_tray_rpy = np.array([0.0, 0.0, 2.111], dtype=np.float64)
        self.start_joint_positions = np.array([0.41, 1.16, -0.79, -0.11, -0.73, 1.77, 0.42], dtype=np.float64)

        # GOAL configuration
        self.goal_tray_pos = np.array([0.65, 0.50, 0.80], dtype=np.float64)
        self.goal_tray_rpy = np.array([0.0, 0.0, -1.47], dtype=np.float64)

        # Cylinder start
        self.start_cylinder = np.array([0.788, 0.108, 0.655], dtype=np.float64)

        # Success criteria and horizons
        self.success_hold_H = 5                # steps to hold at goal
        self.goal_pos_tolerance = 0.03         # meters
        self.goal_yaw_tolerance = 5.0 * np.pi/180.0  # radians (~5 deg)
        self.max_steps = 150                   # truncate if not done by 150

        # Internal counters
        self.t = 0
        self.goal_hold_counter = 0

        # Joint limits
        self.joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Build robust mapping: joint name -> qpos/qvel/dof index and actuator index
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

            # Find actuator that drives this joint
            act_id = None
            for a in range(self.model.nu):
                if self.model.actuator_trnid[a, 0] == jid:
                    act_id = a
                    break
            if act_id is None:
                raise ValueError(f"No actuator found for joint {jname}")
            self.arm_act_ids.append(act_id)

        # Cache IDs
        self.cylinder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
        if self.cylinder_joint == -1:
            raise ValueError("cylinder_free joint not found in MuJoCo model")
        self.cylinder_qposadr = int(self.model.jnt_qposadr[self.cylinder_joint])
        self.cylinder_dofadr = int(self.model.jnt_dofadr[self.cylinder_joint])
        self.cylinder_type = int(self.model.jnt_type[self.cylinder_joint])

        # Control frame: tray_base (the physical tray body)
        self.tray_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_base")
        if self.tray_body_id == -1:
            raise ValueError("Body 'tray_base' not found in MuJoCo model")

        # Joint-space PD gains for torque control (optimized from grid search)
        self.Kq = np.array([2000.0, 2000.0, 2000.0, 1500.0, 500.0, 400.0, 300.0], dtype=np.float64)
        self.Dq = np.array([89.4427191, 89.4427191, 89.4427191, 77.45966692, 44.72135955, 40.0, 34.64101615], dtype=np.float64)
        
        # Torque limits per joint
        self.tau_limits = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)

        # Desired joint target (initialized at start pose)
        self.q_des = self.start_joint_positions.copy()

        # For velocity estimation (used when not using Jacobian)
        self.prev_tray_pos = None
        self.prev_tray_rpy = None
        self._prev_cyl_pos = None
        
        # For Jacobian-based velocity estimation
        self.prev_joint_pos_noisy = None
        self.prev_joint_vel_noisy = None

        print(f"\n{'='*60}")
        print(f"TrayPoseEnv initialized with JOINT-SPACE TORQUE CONTROL")
        print(f"  Control frame: tray_base (body_id={self.tray_body_id})")
        print(f"  Joint-space gains: Kq={self.Kq}")
        print(f"  Damping gains: Dq={self.Dq}")
        print(f"  Action space: 7D joint increments")
        print(f"  Max joint increment: {self.max_joint_increment} rad/step")
        print(f"  Sticky q_des: enabled (zero action holds last target)")
        print(f"  Observation noise: pos_std={self.obs_noise_std_pos:.4f} rad, vel_std={self.obs_noise_std_vel:.4f} rad/s")
        print(f"  Tray obs mode: {'Jacobian FK' if self.use_jacobian_tray_obs else 'Direct MuJoCo'}")
        print(f"  Observation space: 34D (includes cylinder xy velocity in tray frame)")
        print(f"  Success criteria: hold goal for {self.success_hold_H} steps")
        print(f"  Goal tolerances: pos={self.goal_pos_tolerance}m, yaw={self.goal_yaw_tolerance*180/np.pi:.1f}°")
        print(f"  Max episode steps: {self.max_steps}")
        print("\nActuator -> Joint -> DOF mapping:")
        for i in range(7):
            act_name_start = self.model.name_actuatoradr[self.arm_act_ids[i]]
            act_name = self.model.names[act_name_start:].split(b'\x00')[0].decode('utf-8')
            gear = self.model.actuator_gear[self.arm_act_ids[i], 0]
            print(f"  [{i}] actuator={self.arm_act_ids[i]:2d} {act_name:20s} -> "
                  f"joint={self.arm_jnt_ids[i]:2d} {self.arm_joint_names[i]:15s} | "
                  f"qposadr={self.arm_qposadr[i]:2d} dofadr={self.arm_dofadr[i]:2d} | gear={gear:.1f}")
        print(f"{'='*60}\n")

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _get_arm_qpos(self, noisy=False):
        """
        Get current joint positions in arm joint order.
        
        Args:
            noisy: If True, add Gaussian noise to simulate sensor noise
        """
        q = np.array([self.data.qpos[addr] for addr in self.arm_qposadr], dtype=np.float64)
        if noisy and self.obs_noise_std_pos > 0:
            q += np.random.normal(0, self.obs_noise_std_pos, size=q.shape)
        return q

    def _get_arm_qvel(self, noisy=False):
        """
        Get current joint velocities in arm joint order.
        
        Args:
            noisy: If True, add Gaussian noise to simulate sensor noise
        """
        qd = np.array([self.data.qvel[addr] for addr in self.arm_dofadr], dtype=np.float64)
        if noisy and self.obs_noise_std_vel > 0:
            qd += np.random.normal(0, self.obs_noise_std_vel, size=qd.shape)
        return qd

    def _set_arm_qpos(self, q):
        """Set arm joint positions (for reset only)."""
        for i, addr in enumerate(self.arm_qposadr):
            self.data.qpos[addr] = float(q[i])

    def _set_arm_ctrl(self, tau):
        """Write torques to arm actuators."""
        for i, act_id in enumerate(self.arm_act_ids):
            self.data.ctrl[act_id] = float(tau[i])

    def _compute_tray_fk_jacobian(self, joint_pos, joint_vel):
        """
        Compute tray_base pose and velocity using Jacobian-based FK from joint states.
        
        Args:
            joint_pos: 7D joint positions (potentially noisy)
            joint_vel: 7D joint velocities (potentially noisy)
        
        Returns:
            pos: 3D position
            rpy: 3D roll-pitch-yaw
            linear_vel: 3D linear velocity
            angular_vel: 3D angular velocity (in body frame)
        """
        # Create temporary data structure for FK computation
        temp_data = mujoco.MjData(self.model)
        
        # Set joint positions
        for i, addr in enumerate(self.arm_qposadr):
            temp_data.qpos[addr] = joint_pos[i]
        
        # Set joint velocities
        for i, addr in enumerate(self.arm_dofadr):
            temp_data.qvel[addr] = joint_vel[i]
        
        # Compute forward kinematics
        mujoco.mj_forward(self.model, temp_data)
        
        # Extract tray_base pose
        pos = temp_data.xpos[self.tray_body_id].copy()
        quat_wxyz = temp_data.xquat[self.tray_body_id].copy()  # [w,x,y,z]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rpy = R.from_quat(quat_xyzw).as_euler('xyz')
        
        # Compute Jacobian for velocity
        jacp = np.zeros((3, self.model.nv))  # position Jacobian
        jacr = np.zeros((3, self.model.nv))  # rotation Jacobian
        mujoco.mj_jacBody(self.model, temp_data, jacp, jacr, self.tray_body_id)
        
        # Extract columns for arm DOFs
        jacp_arm = jacp[:, self.arm_dofadr]  # 3x7
        jacr_arm = jacr[:, self.arm_dofadr]  # 3x7
        
        # Compute velocities: v = J * q_dot
        linear_vel = jacp_arm @ joint_vel
        angular_vel = jacr_arm @ joint_vel
        
        return pos, rpy, linear_vel, angular_vel

    def _get_tray_pose_velocity(self, use_noisy_joints=False):
        """
        Get tray pose and velocity either from MuJoCo directly or via Jacobian FK.
        
        Args:
            use_noisy_joints: If True and using Jacobian mode, use noisy joint states
        
        Returns:
            pos: 3D position
            rpy: 3D roll-pitch-yaw
            linear_vel: 3D linear velocity
            angular_vel: 3D angular velocity
        """
        if self.use_jacobian_tray_obs:
            # Use Jacobian-based FK with potentially noisy joint states
            joint_pos = self._get_arm_qpos(noisy=use_noisy_joints)
            joint_vel = self._get_arm_qvel(noisy=use_noisy_joints)
            return self._compute_tray_fk_jacobian(joint_pos, joint_vel)
        else:
            # Use direct MuJoCo body pose (ground truth)
            pos = self.data.xpos[self.tray_body_id].copy()
            quat_wxyz = self.data.xquat[self.tray_body_id].copy()
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rpy = R.from_quat(quat_xyzw).as_euler('xyz')
            
            # Get velocities from MuJoCo if available, else finite difference
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
            
            # Fallback to finite difference if needed
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

    def _get_cylinder_world_pos_vel(self):
        """
        Get cylinder world position and velocity.
        Returns: (pos, vel) both as 3D numpy arrays in world frame.
        """
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            pos = self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
            vel = self.data.qvel[self.cylinder_dofadr:self.cylinder_dofadr+3].copy()  # linear vel in world
        else:
            # Fallbacks: try body named cylinder or finite difference
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
                pos = self.data.xpos[bid].copy()
                # If xvelp available, use it; else finite-diff
                if hasattr(self.data, "xvelp"):
                    vel = self.data.xvelp[bid].copy()
                else:
                    # finite difference
                    if self._prev_cyl_pos is None:
                        vel = np.zeros(3)
                    else:
                        vel = (pos - self._prev_cyl_pos) / self.control_dt
                    self._prev_cyl_pos = pos.copy()
            except Exception:
                # As a last resort, use qpos slice if present
                try:
                    pos = self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
                except Exception:
                    pos = np.zeros(3)
                vel = np.zeros(3)
        return pos, vel

    def reset(self):
        self.t = 0
        self.goal_hold_counter = 0
        mujoco.mj_resetData(self.model, self.data)

        # Set robot state to start pose
        self._set_arm_qpos(self.start_joint_positions)
        for dof in self.arm_dofadr:
            self.data.qvel[dof] = 0.0

        # Initialize desired joint target
        self.q_des = self.start_joint_positions.copy()

        # Zero all controls
        self.data.ctrl[:] = 0.0

        # Place the cylinder
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+7] = np.array(
                [*self.start_cylinder, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.data.qvel[self.cylinder_dofadr:self.cylinder_dofadr+6] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_SLIDE:
            self.data.qpos[self.cylinder_qposadr] = float(self.start_cylinder[2])
            self.data.qvel[self.cylinder_dofadr] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_HINGE:
            self.data.qpos[self.cylinder_qposadr] = 0.0
            self.data.qvel[self.cylinder_dofadr] = 0.0
        else:
            try:
                self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3] = np.array(self.start_cylinder, dtype=np.float64)
                self.data.qvel[self.cylinder_dofadr:self.cylinder_dofadr+3] = 0.0
            except Exception:
                pass

        # Realize state
        mujoco.mj_forward(self.model, self.data)

        # Record tray pose (ground truth for control)
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()  # [w,x,y,z]
        tray_quat_xyzw = [tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]
        tray_rpy = R.from_quat(tray_quat_xyzw).as_euler('xyz')
        self.tray_yaw = tray_rpy[2]
        self.tray_rpy = tray_rpy.copy()

        # Reset history for finite difference
        self.prev_tray_pos = self.tray_pos.copy()
        self.prev_tray_rpy = tray_rpy.copy()
        self._prev_cyl_pos = None
        self.prev_joint_pos_noisy = None
        self.prev_joint_vel_noisy = None

        return self._get_obs()

    def _compute_torque_control(self):
        """
        Joint-space PD with gravity compensation:
        tau = Kq*(q_des - q) - Dq*qdot + qfrc_bias
        
        NOTE: Uses clean (non-noisy) joint states for control to simulate
        that the low-level controller has access to better state estimates.
        """
        q = self._get_arm_qpos(noisy=False)  # Control uses clean state
        qd = self._get_arm_qvel(noisy=False)
        
        # PD error
        err = self.q_des - q
        tau = self.Kq * err - self.Dq * qd

        # Add gravity/Coriolis/centrifugal compensation from the correct DOFs
        tau += np.array([self.data.qfrc_bias[dof] for dof in self.arm_dofadr], dtype=np.float64)

        # Clip to actuator limits
        tau = np.clip(tau, -self.tau_limits, self.tau_limits)
        
        return tau

    def step(self, action):
        # Clip action and integrate
        action = np.clip(action, -1.0, 1.0)
        self.t += 1

        # Convert action to joint target increments with sticky target
        dq_cmd = action * self.max_joint_increment
        q = self._get_arm_qpos(noisy=False)
        if not np.allclose(action, 0.0, atol=1e-6):
            self.q_des = np.clip(q + dq_cmd, self.joint_limits_low, self.joint_limits_high)

        # Physics substeps with torque control
        for _ in range(self.substeps):
            tau = self._compute_torque_control()
            self._set_arm_ctrl(tau)
            mujoco.mj_step(self.model, self.data)

        # Update tray pose (ground truth for reward/termination)
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()
        self.tray_rpy = R.from_quat([tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]).as_euler('xyz')
        self.tray_yaw = self.tray_rpy[2]

        # Build observation
        obs = self._get_obs()

        # Unpack for rewards/termination
        cyl_pos_w, _ = self._get_cylinder_world_pos_vel()

        # Compute goal proximity
        yaw_err = self._wrap_angle(self.tray_yaw - self.goal_tray_rpy[2])
        pos_err = np.linalg.norm(self.tray_pos - self.goal_tray_pos)
        at_goal_now = (pos_err <= self.goal_pos_tolerance) and (abs(yaw_err) <= self.goal_yaw_tolerance)

        # Maintain goal hold counter
        if at_goal_now:
            self.goal_hold_counter += 1
        else:
            self.goal_hold_counter = 0

        # Base time penalty every step
        reward = -0.1

        # Extra penalty if agent stays in place (encourage exploration)
        action_mag = float(np.linalg.norm(action))
        if action_mag <= 1e-3:
            reward += -0.4

        # Rim penalty (small)
        rel_cyl_xy_world = cyl_pos_w[:2] - self.tray_pos[:2]
        x_min, x_max = -0.07, 0.07
        y_min, y_max = -0.11, 0.11
        if (rel_cyl_xy_world[0] < x_min or rel_cyl_xy_world[0] > x_max or
            rel_cyl_xy_world[1] < y_min or rel_cyl_xy_world[1] > y_max):
            reward += -0.5

        terminated = False
        truncated = False

        # Drop termination (large penalty)
        drop_threshold = 0.10  # m
        if cyl_pos_w[2] < drop_threshold:
            reward += -10.0
            terminated = True

        # Topple termination (medium penalty)
        if not terminated and self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            cyl_quat = self.data.qpos[self.cylinder_qposadr+3:self.cylinder_qposadr+7]
            cyl_rot = R.from_quat([cyl_quat[1], cyl_quat[2], cyl_quat[3], cyl_quat[0]])
            z_axis = cyl_rot.apply([0, 0, 1])
            angle_from_upright = np.arccos(np.clip(z_axis[2], -1.0, 1.0))
            if angle_from_upright > (np.pi / 4):  # > 45 deg from upright
                reward += -5.0
                terminated = True

        # Success check: if held goal for H steps, grant final positive reward
        if not terminated and self.goal_hold_counter >= self.success_hold_H:
            # Cylinder distance to tray center in tray plane (use world XY relative to tray center)
            cyl_offset = np.linalg.norm(rel_cyl_xy_world)

            # Final positive reward inversely proportional to offset
            # Example: R_final = max(0, 5.0 - alpha * offset)
            alpha = 50.0  # scale factor (1/m)
            final_bonus = max(0.0, 5.0 - alpha * cyl_offset)
            reward += final_bonus
            terminated = True  # task finishes successfully

        # Time limit truncation
        if not terminated and self.t >= self.max_steps:
            truncated = True

        # Return Gymnasium-style tuple (obs, reward, terminated, truncated, info)
        info = {
            'goal_hold_counter': self.goal_hold_counter,
            'at_goal': at_goal_now,
            'pos_err': pos_err,
            'yaw_err': yaw_err,
            'cylinder_offset': np.linalg.norm(rel_cyl_xy_world) if not terminated else 0.0
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def autotune_hold_pose(self,
                           hold_seconds=2.0,
                           max_pos_error_rad=0.01,
                           max_iters=8,
                           verbose=True):
        """
        Adjusts scalar multipliers on Kq to ensure the arm holds pose under gravity with zero action.
        Returns updated Kq, Dq.
        NOTE: With optimized gains already set, this is optional and mainly for testing new loads.
        """
        base_Kq = self.Kq.copy()
        # Two group multipliers: proximal (j1-4), distal (j5-7)
        m_prox = 1.0
        m_dist = 1.0

        def apply_gains(m1, m2):
            K = base_Kq.copy()
            K[:4] *= m1
            K[4:] *= m2
            D = 2.0 * np.sqrt(K)
            return K, D

        # Utility to run zero-action hold and measure errors and saturation
        def eval_hold(K, D):
            # Backup and set gains
            K_backup, D_backup = self.Kq.copy(), self.Dq.copy()
            self.Kq, self.Dq = K.copy(), D.copy()

            # Reset to start pose
            self.reset()
            q_ref = self._get_arm_qpos(noisy=False).copy()

            n_steps = int(hold_seconds / self.control_dt)
            max_err = 0.0
            sat_count = np.zeros(7, dtype=int)

            for _ in range(n_steps):
                # zero action
                dq_cmd = np.zeros(7, dtype=np.float64)
                q = self._get_arm_qpos(noisy=False)
                self.q_des = np.clip(q + dq_cmd, self.joint_limits_low, self.joint_limits_high)

                # substeps
                for _ in range(self.substeps):
                    # compute raw tau without clipping, to check saturation
                    q_now = self._get_arm_qpos(noisy=False)
                    qd_now = self._get_arm_qvel(noisy=False)
                    err = self.q_des - q_now
                    tau = self.Kq * err - self.Dq * qd_now
                    tau += np.array([self.data.qfrc_bias[d] for d in self.arm_dofadr], dtype=np.float64)

                    # Count saturation if out of range
                    clipped = np.clip(tau, -self.tau_limits, self.tau_limits)
                    sat_count += (np.abs(tau) > self.tau_limits + 1e-6).astype(int)

                    self._set_arm_ctrl(clipped)
                    mujoco.mj_step(self.model, self.data)

                # track error
                max_err = max(max_err, float(np.max(np.abs(self._get_arm_qpos(noisy=False) - q_ref))))

            # Restore backups for safety (caller will set final gains)
            self.Kq, self.Dq = K_backup, D_backup

            return max_err, sat_count

        for it in range(max_iters):
            K_try, D_try = apply_gains(m_prox, m_dist)
            max_err, sat_count = eval_hold(K_try, D_try)

            if verbose:
                print(f"[autotune {it}] m_prox={m_prox:.2f} m_dist={m_dist:.2f} "
                      f"max_err={max_err*180/np.pi:.3f} deg, sat={sat_count}")

            # Success condition
            if max_err <= max_pos_error_rad:
                self.Kq, self.Dq = K_try, D_try
                if verbose:
                    print(f"✓ Hold-pose achieved. Final Kq={self.Kq}, Dq={self.Dq}")
                return self.Kq.copy(), self.Dq.copy()

            # If proximal joints saturate often, try increasing proximal gain carefully up to the point before saturation dominates
            prox_sat = np.sum(sat_count[:4])  # total saturation events across time and joints 1-4
            dist_sat = np.sum(sat_count[4:])  # joints 5-7

            # Heuristic update:
            # - If not saturating much, increase gains for the group contributing most to error (usually proximal).
            # - If saturating a lot, back off that group's multiplier slightly.
            if prox_sat < 5:
                m_prox *= 1.3
            else:
                m_prox *= 0.9

            if dist_sat < 5:
                m_dist *= 1.2
            else:
                m_dist *= 0.9

            # Clamp multipliers to reasonable bounds
            m_prox = float(np.clip(m_prox, 0.5, 5.0))
            m_dist = float(np.clip(m_dist, 0.5, 5.0))

        # If we exit without success, set the best try anyway
        self.Kq, self.Dq = apply_gains(m_prox, m_dist)
        if verbose:
            print(f"⚠ Autotune reached max iterations; set last gains. Final Kq={self.Kq}, Dq={self.Dq}")
            print("Consider reducing tray mass or increasing torque limits (if realistic).")
        return self.Kq.copy(), self.Dq.copy()

    def _get_cylinder_xyz(self):
        """Legacy helper for backward compatibility - returns only position."""
        pos, _ = self._get_cylinder_world_pos_vel()
        return pos

    def _get_obs(self):
        """
        Build observation vector (34D):
        - Cylinder state in tray frame: [x, y] (2)
        - Tray pose: [x, y, z, r, p, y] (6)
        - Tray velocities: [vx, vy, vz, wx, wy, wz] (6)
        - Joint angles (7)
        - Joint velocities (7)
        - Goal pose: [x, y, z, yaw] (4)
        - Cylinder velocity in tray frame: [vx, vy] (2)
        """
        # Get tray pose and velocity (either from MuJoCo or Jacobian FK)
        current_tray_pos, tray_rpy, tray_linear_velocity, tray_angular_velocity = self._get_tray_pose_velocity(use_noisy_joints=True)
        
        # Convert RPY to quaternion for rotation matrix
        tray_quat_xyzw = R.from_euler('xyz', tray_rpy).as_quat()

        # Cylinder world position and velocity
        cyl_pos_w, cyl_vel_w = self._get_cylinder_world_pos_vel()

        # Transform cylinder pos/vel to tray frame
        R_tray_w = R.from_quat(tray_quat_xyzw).as_matrix()      # world <- tray
        R_w_tray = R_tray_w.T                                   # tray <- world
        # Relative position in world then to tray frame
        rel_pos_w = cyl_pos_w - current_tray_pos
        rel_pos_tray = R_w_tray @ rel_pos_w
        # Relative velocity: cylinder vel minus tray linear vel, then rotate to tray frame
        rel_vel_w = cyl_vel_w - tray_linear_velocity
        rel_vel_tray = R_w_tray @ rel_vel_w
        # Extract planar components
        cyl_xy_in_tray = rel_pos_tray[:2]
        cyl_vxy_in_tray = rel_vel_tray[:2]

        # Joint states (with noise for observation)
        joint_angles = self._get_arm_qpos(noisy=True)
        joint_velocities = self._get_arm_qvel(noisy=True)

        # Goal
        goal_pose = np.concatenate([self.goal_tray_pos, np.array([self.goal_tray_rpy[2]])])

        return np.concatenate([
            cyl_xy_in_tray,              # 2: cylinder x,y in tray frame
            current_tray_pos,            # 3
            tray_rpy,                    # 3
            tray_linear_velocity,        # 3
            tray_angular_velocity,       # 3
            joint_angles,                # 7
            joint_velocities,            # 7
            goal_pose,                   # 4
            cyl_vxy_in_tray              # 2: cylinder xdot, ydot in tray frame
        ]).astype(np.float32)

    def render(self, mode="human"):
        cylinder_state = self._get_cylinder_xyz()
        print(f"Step {self.t}: Tray {self.tray_pos}, Cylinder {cylinder_state}")

    def close(self):
        pass