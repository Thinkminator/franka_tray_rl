import gym
import numpy as np
import mujoco
from gym import spaces
from scipy.spatial.transform import Rotation as R
import pybullet as p
import pybullet_data

class TrayPoseEnv(gym.Env):
    """
    Custom environment for tray-cylinder manipulation (traypose variant).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path="assets/panda_tray/panda_tray_cylinder.xml",
                 kp=50.0, kd=2.0, max_joint_increment=0.01):
        super(TrayPoseEnv, self).__init__()

        # Load Mujoco model (combined robot + tray + cylinder)
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Action: 7-DOF normalized joint position increments [-1, 1]
        self.control_dt = 0.02
        self.substeps = 5  # Physics substeps for smoother motion
        
        # PD controller parameters
        self.kp = float(kp)  # proportional gain
        self.kd = float(kd)   # derivative gain
        self.max_joint_increment = float(max_joint_increment)  # max joint position increment per step (radians)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation: 32 dimensions (enhanced)
        obs_dim = 32
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define start & goal poses
        self.start_tray_pos = np.array([0.5, 0.3, 0.65])
        self.start_tray_rpy = np.array([0.0, 0.0, -3.07])
        self.goal_tray_pos = np.array([0.5, -0.35, 0.2])
        self.goal_tray_rpy = np.array([0.0, 0.0, -4.57])

        self.start_cylinder = np.array([0.5, 0.3, 0.7])

        self.t = 0
        self.max_steps = 500

        # Cache Mujoco IDs
        self.cylinder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
        self.cylinder_qposadr = self.model.jnt_qposadr[self.cylinder_joint]
        self.cylinder_type = self.model.jnt_type[self.cylinder_joint]
        
        # Get tray body ID
        self.tray_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tray_base")
        if self.tray_body_id == -1:
            raise ValueError("Body 'tray_base' not found in model. Check XML body names.")

        # For velocity estimation
        self.prev_tray_pos = None
        self.prev_tray_rpy = None

        # Initialize PyBullet for IK (run once)
        self._init_pybullet()

    def _init_pybullet(self):
        """Initialize PyBullet for IK calculations"""
        p.connect(p.DIRECT)  # Run headless
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load URDF (using your actual URDF without cylinder)
        urdf_path = "assets/panda_tray/panda_tray.urdf"
        self.pb_robot = p.loadURDF(urdf_path, useFixedBase=True)
        
        # Find tray base link index
        num_joints = p.getNumJoints(self.pb_robot)
        self.tray_link_index = -1
        for i in range(num_joints):
            if p.getJointInfo(self.pb_robot, i)[12].decode("utf-8") == "tray_base":
                self.tray_link_index = i
                break
        
        if self.tray_link_index == -1:
            raise ValueError("Tray base link not found in PyBullet model")
            
        # Get movable joints for IK
        self.pb_movable_joints = [i for i in range(num_joints) if p.getJointInfo(self.pb_robot, i)[2] != p.JOINT_FIXED]

    def _compute_ik(self, tray_pos, tray_rpy):
        """Compute IK using PyBullet for desired tray pose"""
        target_quat = p.getQuaternionFromEuler(tray_rpy)
        
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.pb_robot,
            endEffectorLinkIndex=self.tray_link_index,
            targetPosition=tray_pos,
            targetOrientation=target_quat,
            maxNumIterations=200,
            residualThreshold=1e-4
        )
        
        return joint_positions[:7]  # Return first 7 joint values

    def _set_mujoco_joints_from_ik(self, ik_joint_values):
        """Map PyBullet IK joint values to MuJoCo qpos"""
        for idx, pb_joint_idx in enumerate(self.pb_movable_joints):
            if idx >= len(ik_joint_values):
                break
            pb_name = p.getJointInfo(self.pb_robot, pb_joint_idx)[1].decode("utf-8")
            ik_value = ik_joint_values[idx]

            try:
                mj_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, pb_name)
            except Exception:
                mj_jid = -1

            if mj_jid != -1:
                qposadr = int(self.model.jnt_qposadr[mj_jid])
                self.data.qpos[qposadr] = ik_value

    def reset(self):
        self.t = 0

        ik_joints = self._compute_ik(self.start_tray_pos, self.start_tray_rpy)
        self._set_mujoco_joints_from_ik(ik_joints)

        for i in range(7):
            self.data.qvel[i] = 0.0

        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+7] = np.array(
                [*self.start_cylinder, 1, 0, 0, 0]
            )
            self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+6] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_SLIDE:
            self.data.qpos[self.cylinder_qposadr] = self.start_cylinder[2]
            self.data.qvel[self.cylinder_qposadr] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_HINGE:
            self.data.qpos[self.cylinder_qposadr] = 0.0
            self.data.qvel[self.cylinder_qposadr] = 0.0
        else:
            try:
                self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3] = np.array(self.start_cylinder)
                self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+3] = 0.0
            except Exception:
                pass

        if hasattr(self.data, "ctrl"):
            self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()
        tray_quat_xyzw = [tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]
        tray_rpy = R.from_quat(tray_quat_xyzw).as_euler('xyz')
        self.tray_yaw = tray_rpy[2]
        self.tray_rpy = tray_rpy.copy()

        self.prev_tray_pos = self.tray_pos.copy()
        self.prev_tray_rpy = tray_rpy.copy()

        return self._get_obs()

    def step(self, action):
        self.t += 1

        # Clip action to [-1,1]
        action = np.clip(action, -1.0, 1.0)

        # Map action to joint position increments
        delta_q = action * self.max_joint_increment

        # Compute target joint positions
        target_qpos = self.data.qpos[:7] + delta_q

        # Enforce joint limits on target positions
        joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        target_qpos = np.clip(target_qpos, joint_limits_low, joint_limits_high)

        # Current joint positions and velocities
        q = self.data.qpos[:7]
        qd = self.data.qvel[:7]

        # PD control torque
        tau = self.kp * (target_qpos - q) - self.kd * qd

        # Apply torque to actuators (assuming first 7 actuators correspond to joints)
        self.data.ctrl[:7] = tau

        # Run physics substeps
        for _ in range(self.substeps):
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        # Update tray pose (exact)
        self.tray_pos = self.data.xpos[self.tray_body_id].copy()
        tray_quat = self.data.xquat[self.tray_body_id].copy()
        tray_rpy = R.from_quat([tray_quat[1], tray_quat[2], tray_quat[3], tray_quat[0]]).as_euler('xyz')
        self.tray_yaw = tray_rpy[2]
        self.tray_rpy = tray_rpy.copy()

        obs = self._get_obs()

        # Compute enhanced reward
        cylinder_state = self._get_cylinder_xyz()
        tray_dist = np.linalg.norm(self.tray_pos - self.goal_tray_pos) + abs(self.tray_yaw - self.goal_tray_rpy[2])
        cylinder_offset = np.linalg.norm(cylinder_state[:2] - self.tray_pos[:2])
        
        # Action penalty
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 1e-3:
            action_penalty = -0.1
        else:
            action_penalty = -0.5
        
        # Cylinder rim penalty
        rel_cyl_xy = cylinder_state[:2] - self.tray_pos[:2]
        x_min, x_max = -0.07, 0.07
        y_min, y_max = -0.11, 0.11
        
        if (rel_cyl_xy[0] < x_min or rel_cyl_xy[0] > x_max or
            rel_cyl_xy[1] < y_min or rel_cyl_xy[1] > y_max):
            hit_rim_penalty = -0.5
        else:
            hit_rim_penalty = 0.0
        
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
        if not hasattr(self, 'hold_counter'):
            self.hold_counter = 0
            
        if tray_dist < 0.05:
            self.hold_counter += 1
        else:
            self.hold_counter = 0
        
        if self.hold_counter >= H:
            if cylinder_offset < 0.05:
                positive_reward = 5.0
            else:
                positive_reward = 3.0
        else:
            positive_reward = 0.0
        
        reward = action_penalty + hit_rim_penalty + drop_penalty + topple_penalty + positive_reward
        
        # Termination
        done = done or self.t >= self.max_steps

        return obs, reward, done, {}

    def _get_cylinder_xyz(self):
        """Extracts cylinder XYZ robustly depending on joint type"""
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            return self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3]
        elif self.cylinder_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            # put cylinder at tray position +0z as fallback
            return np.array([*self.start_cylinder])
        else:
            try:
                return self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3]
            except Exception:
                return np.zeros(3)

    def _get_obs(self):
        """Enhanced observation space with 32 dimensions"""
        cylinder_state = self._get_cylinder_xyz()
        
        # Get actual tray pose from simulation
        current_tray_pos = self.data.xpos[self.tray_body_id].copy()
        
        # Use exact RPY stored from reset/step
        tray_rpy = self.tray_rpy.copy()
        
        # Get tray velocities (try direct access, fallback to finite difference)
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

        # Finite-difference fallback
        if tray_linear_velocity is None or tray_angular_velocity is None:
            if self.prev_tray_pos is None:
                tray_linear_velocity = np.zeros(3)
                tray_angular_velocity = np.zeros(3)
            else:
                # Linear velocity
                tray_linear_velocity = (current_tray_pos - self.prev_tray_pos) / self.control_dt

                # Angular velocity (handle wrap-around)
                def angle_diff(a, b):
                    d = (a - b + np.pi) % (2 * np.pi) - np.pi
                    return d

                delta_rpy = np.array([
                    angle_diff(tray_rpy[0], self.prev_tray_rpy[0]),
                    angle_diff(tray_rpy[1], self.prev_tray_rpy[1]),
                    angle_diff(tray_rpy[2], self.prev_tray_rpy[2])
                ])
                tray_angular_velocity = delta_rpy / self.control_dt

        # Update previous pose
        self.prev_tray_pos = current_tray_pos.copy()
        self.prev_tray_rpy = tray_rpy.copy()
        
        # Get robot joint states (actual values)
        joint_angles = np.array([self.data.qpos[i] for i in range(7)])
        joint_velocities = np.array([self.data.qvel[i] for i in range(7)])
        
        # Build enhanced observation (32 dimensions)
        rel_cylinder_pos = (cylinder_state[:2] - current_tray_pos[:2])  # 2D
        goal_pose = np.concatenate([self.goal_tray_pos, np.array([self.goal_tray_rpy[2]])])  # 4D
        return np.concatenate([
            rel_cylinder_pos,              # 2: Relative cylinder XY position
            current_tray_pos,              # 3: Tray XYZ position
            tray_rpy,                      # 3: Tray roll, pitch, yaw
            tray_linear_velocity,          # 3: Tray linear velocity
            tray_angular_velocity,         # 3: Tray angular velocity
            joint_angles,                  # 7: Joint angles
            joint_velocities,              # 7: Joint velocities
            goal_pose                      # 4: Goal XYZ + yaw
        ])  # Total: 32 dimensions

    def render(self, mode="human"):
        cylinder_state = self._get_cylinder_xyz()
        print(f"Step {self.t}: Tray {self.tray_pos}, Cylinder {cylinder_state}")

    def close(self):
        try:
            p.disconnect()
        except:
            pass

    def set_pd(self, kp=None, kd=None, max_joint_increment=None):
        """Update PD gains and max increment at runtime."""
        if kp is not None:
            self.kp = float(kp)
        if kd is not None:
            self.kd = float(kd)
        if max_joint_increment is not None:
            self.max_joint_increment = float(max_joint_increment)
        # optional: sanity bounds check
        # e.g., ensure non-negative
        if self.kp < 0 or self.kd < 0 or self.max_joint_increment <= 0:
            raise ValueError("kp/kd must be >= 0 and max_joint_increment > 0")