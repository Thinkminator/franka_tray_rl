import gym
import numpy as np
import mujoco
from gym import spaces
from scipy.spatial.transform import Rotation as R

class TrayPoseEnv(gym.Env):
    """
    Custom environment for tray-cylinder manipulation (traypose variant).
    Uses tray_frame as the controlled/observed tray pose.
    Direct joint position control with normalized actions.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path="assets/panda_tray/panda_tray_cylinder.xml",
                 max_joint_increment=0.05):
        super(TrayPoseEnv, self).__init__()

        # Load Mujoco model (combined robot + tray + cylinder)
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control settings
        self.control_dt = 0.02
        self.substeps = 5  # Physics substeps for smoother motion
        
        # Direct position control parameters
        self.max_joint_increment = float(max_joint_increment)  # rad per step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation: 32 dimensions
        obs_dim = 32
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ===== START configuration (from your interactive session) =====
        # Use achieved pose for reference
        self.start_tray_pos = np.array([0.788, 0.108, 0.621], dtype=np.float64)
        self.start_tray_rpy = np.array([0.0, 0.0, 2.111], dtype=np.float64)
        self.start_joint_positions = np.array([0.41, 1.16, -0.79, -0.11, -0.73, 1.77, 0.42], dtype=np.float64)

        # ===== GOAL configuration (from your interactive session) =====
        self.goal_tray_pos = np.array([0.65, 0.50, 0.80], dtype=np.float64)
        self.goal_tray_rpy = np.array([0.0, 0.0, -1.47], dtype=np.float64)

        # Cylinder initial placement
        self.start_cylinder = np.array([1.0, 0.0, 0.70], dtype=np.float64)

        self.t = 0
        self.max_steps = 500

        # Joint limits for Panda robot
        self.joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Cache Mujoco IDs
        self.cylinder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
        self.cylinder_qposadr = self.model.jnt_qposadr[self.cylinder_joint]
        self.cylinder_type = self.model.jnt_type[self.cylinder_joint]
        
        # Validate cylinder joint
        assert self.cylinder_joint != -1, "cylinder_free joint not found in MuJoCo model"
        assert self.cylinder_qposadr >= 0, "Invalid cylinder joint qpos address"
        
        # Use tray_frame body in MuJoCo
        self.tray_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_frame")
        if self.tray_body_id == -1:
            raise ValueError("Body 'tray_frame' not found in MuJoCo model. Check XML body names.")

        # For velocity estimation
        self.prev_tray_pos = None
        self.prev_tray_rpy = None

    @staticmethod
    def _wrap_angle(a):
        """Wrap angle to [-pi, pi]"""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _set_mujoco_joints_from_list(self, joint_values):
        """Directly set MuJoCo joint qpos by Panda joint names"""
        joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for i, name in enumerate(joint_names):
            mj_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if mj_jid != -1:
                qposadr = int(self.model.jnt_qposadr[mj_jid])
                self.data.qpos[qposadr] = float(joint_values[i])

    def reset(self):
        self.t = 0
        self.hold_counter = 0

        # Full data reset - clears all state including contacts and constraints
        mujoco.mj_resetData(self.model, self.data)

        # Seed the robot to provided joint configuration
        self._set_mujoco_joints_from_list(self.start_joint_positions)
        self.data.qvel[:7] = 0.0

        # Place the cylinder (robustly for free joint)
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+7] = np.array(
                [*self.start_cylinder, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+6] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_SLIDE:
            self.data.qpos[self.cylinder_qposadr] = float(self.start_cylinder[2])
            self.data.qvel[self.cylinder_qposadr] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_HINGE:
            self.data.qpos[self.cylinder_qposadr] = 0.0
            self.data.qvel[self.cylinder_qposadr] = 0.0
        else:
            # Best-effort 3 DOF
            try:
                self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3] = np.array(self.start_cylinder, dtype=np.float64)
                self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+3] = 0.0
            except Exception:
                pass

        # Clear controls
        if hasattr(self.data, "ctrl"):
            self.data.ctrl[:] = 0.0

        # Forward to realize state
        mujoco.mj_forward(self.model, self.data)

        # Record actual tray_frame pose from MuJoCo
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()  # [w,x,y,z]
        tray_quat_xyzw = [tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]
        tray_rpy = R.from_quat(tray_quat_xyzw).as_euler('xyz')
        self.tray_yaw = tray_rpy[2]
        self.tray_rpy = tray_rpy.copy()

        self.prev_tray_pos = self.tray_pos.copy()
        self.prev_tray_rpy = tray_rpy.copy()

        return self._get_obs()

    def step(self, action):
        self.t += 1

        # Clip action to [-1,1] (normalized)
        action = np.clip(action, -1.0, 1.0)

        # Map normalized action to joint position increments
        delta_q = action * self.max_joint_increment * self.control_dt

        # Compute target joint positions
        current_qpos = self.data.qpos[:7].copy()
        target_qpos = current_qpos + delta_q

        # Enforce joint limits on target positions
        target_qpos = np.clip(target_qpos, self.joint_limits_low, self.joint_limits_high)

        # Directly set joint positions (position control)
        self._set_mujoco_joints_from_list(target_qpos)

        # Run physics substeps
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        # Update tray_frame pose (exact from MuJoCo)
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()
        self.tray_rpy = R.from_quat([tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]).as_euler('xyz')
        self.tray_yaw = self.tray_rpy[2]

        obs = self._get_obs()

        # Compute reward
        cylinder_state = self._get_cylinder_xyz()
        
        # Goal distance with wrapped yaw
        yaw_err = self._wrap_angle(self.tray_yaw - self.goal_tray_rpy[2])
        tray_dist = np.linalg.norm(self.tray_pos - self.goal_tray_pos) + abs(yaw_err)
        cylinder_offset = np.linalg.norm(cylinder_state[:2] - self.tray_pos[:2])
        
        # Action penalty
        action_magnitude = np.linalg.norm(action)
        action_penalty = -0.1 if action_magnitude > 1e-3 else -0.5
        
        # Cylinder rim penalty
        rel_cyl_xy = cylinder_state[:2] - self.tray_pos[:2]
        x_min, x_max = -0.07, 0.07
        y_min, y_max = -0.11, 0.11
        hit_rim_penalty = -0.5 if (rel_cyl_xy[0] < x_min or rel_cyl_xy[0] > x_max or
                                   rel_cyl_xy[1] < y_min or rel_cyl_xy[1] > y_max) else 0.0
        
        # Drop penalty
        drop_threshold = 0.1
        drop_penalty = 0.0
        done = False
        if cylinder_state[2] < drop_threshold:
            drop_penalty = -10.0
            done = True
        
        # Topple penalty
        topple_penalty = 0.0
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            cyl_quat = self.data.qpos[self.cylinder_qposadr+3:self.cylinder_qposadr+7]
            cyl_rot = R.from_quat([cyl_quat[1], cyl_quat[2], cyl_quat[3], cyl_quat[0]])
            z_axis = cyl_rot.apply([0, 0, 1])
            angle_from_upright = np.arccos(np.clip(z_axis[2], -1.0, 1.0))
            if angle_from_upright > np.pi / 4:
                topple_penalty = -5.0
                done = True
        
        # Goal reward
        H = 20
        if tray_dist < 0.05:
            self.hold_counter += 1
        else:
            self.hold_counter = 0
        
        if self.hold_counter >= H:
            positive_reward = 5.0 if cylinder_offset < 0.05 else 3.0
        else:
            positive_reward = 0.0
        
        reward = action_penalty + hit_rim_penalty + drop_penalty + topple_penalty + positive_reward
        
        # Termination
        done = done or self.t >= self.max_steps

        return obs, reward, done, {}

    def _get_cylinder_xyz(self):
        """Extracts cylinder XYZ robustly depending on joint type"""
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
        elif self.cylinder_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            return np.array([*self.start_cylinder])
        else:
            try:
                return self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3].copy()
            except Exception:
                return np.zeros(3)

    def _get_obs(self):
        """Enhanced observation space with 32 dimensions (using tray_frame)"""
        cylinder_state = self._get_cylinder_xyz()
        
        # Get actual tray_frame pose from simulation
        current_tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_rpy = self.tray_rpy.copy()  # computed in reset/step
        
        # Velocities (direct if available, else FD)
        if hasattr(self.data, "xvelp") and hasattr(self.data, "xvelr"):
            try:
                tray_linear_velocity = self.data.xvelp[self.tray_body_id].copy()
                tray_angular_velocity = self.data.xvelr[self.tray_body_id].copy()
            except Exception:
                tray_linear_velocity = None
                tray_angular_velocity = None
        else:
            tray_linear_velocity = None
            tray_angular_velocity = None

        if tray_linear_velocity is None or tray_angular_velocity is None:
            if self.prev_tray_pos is None:
                tray_linear_velocity = np.zeros(3)
                tray_angular_velocity = np.zeros(3)
            else:
                tray_linear_velocity = (current_tray_pos - self.prev_tray_pos) / self.control_dt

                def angle_diff(a, b):
                    return (a - b + np.pi) % (2 * np.pi) - np.pi

                delta_rpy = np.array([
                    angle_diff(tray_rpy[0], self.prev_tray_rpy[0]),
                    angle_diff(tray_rpy[1], self.prev_tray_rpy[1]),
                    angle_diff(tray_rpy[2], self.prev_tray_rpy[2])
                ])
                tray_angular_velocity = delta_rpy / self.control_dt

        # Update previous pose
        self.prev_tray_pos = current_tray_pos.copy()
        self.prev_tray_rpy = tray_rpy.copy()
        
        # Robot joint states
        joint_angles = np.array([self.data.qpos[i] for i in range(7)])
        joint_velocities = np.array([self.data.qvel[i] for i in range(7)])
        
        # Build observation
        rel_cylinder_pos = (cylinder_state[:2] - current_tray_pos[:2])  # 2D
        goal_pose = np.concatenate([self.goal_tray_pos, np.array([self.goal_tray_rpy[2]])])  # 4D
        
        return np.concatenate([
            rel_cylinder_pos,              # 2: Relative cylinder XY position
            current_tray_pos,              # 3: Tray XYZ position (tray_frame)
            tray_rpy,                      # 3: Tray roll, pitch, yaw
            tray_linear_velocity,          # 3: Tray linear velocity
            tray_angular_velocity,         # 3: Tray angular velocity
            joint_angles,                  # 7: Joint angles
            joint_velocities,              # 7: Joint velocities
            goal_pose                      # 4: Goal XYZ + yaw
        ]).astype(np.float32)  # Total: 32 dimensions

    def render(self, mode="human"):
        cylinder_state = self._get_cylinder_xyz()
        print(f"Step {self.t}: TrayFrame {self.tray_pos}, Cylinder {cylinder_state}")

    def close(self):
        """Clean up resources"""
        pass

    def set_max_increment(self, max_joint_increment=None):
        """Update max joint increment at runtime."""
        if max_joint_increment is not None:
            self.max_joint_increment = float(max_joint_increment)
        if self.max_joint_increment <= 0:
            raise ValueError("max_joint_increment must be > 0")