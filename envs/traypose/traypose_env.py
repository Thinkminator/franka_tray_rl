import gym
import numpy as np
import mujoco
from gym import spaces

class TrayPoseEnv(gym.Env):
    """
    Custom environment for tray-cylinder manipulation (traypose variant).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path="assets/panda_tray/panda_tray_cylinder.xml"):
        super(TrayPoseEnv, self).__init__()

        # Load Mujoco model (combined robot + tray + cylinder)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Action: Δtray_x, Δtray_y, Δtray_z, Δyaw
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(4,), dtype=np.float32)

        # Observation: cylinder_rel (3) + tray_pos (3) + tray_yaw (1)
        obs_dim = 3 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define start & goal
        self.start_tray = np.array([0.5, 0.3, 0.65])
        self.start_yaw = -3.07
        self.start_joints = np.array([-0.019, 1.424, -0.833, 1.304, 0.174, 1.803, -0.004])

        self.goal_tray = np.array([0.5, -0.35, 0.2])
        self.goal_yaw = -4.57

        self.start_cylinder = np.array([0.5, 0.3, 0.7])

        self.t = 0
        self.max_steps = 500

        # Cache Mujoco IDs for cylinder
        self.cylinder_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free")
        self.cylinder_qposadr = self.model.jnt_qposadr[self.cylinder_joint]
        self.cylinder_type = self.model.jnt_type[self.cylinder_joint]

    def reset(self):
        self.t = 0

        # Reset robot joints
        for i, q in enumerate(self.start_joints):
            self.data.qpos[i] = q

        # Reset cylinder depending on joint type
        if self.cylinder_type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+7] = np.array(
                [*self.start_cylinder, 1, 0, 0, 0]  # xyz + quat
            )
            self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+6] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_SLIDE:
            self.data.qpos[self.cylinder_qposadr] = self.start_cylinder[2]  # only z
            self.data.qvel[self.cylinder_qposadr] = 0.0
        elif self.cylinder_type == mujoco.mjtJoint.mjJNT_HINGE:
            self.data.qpos[self.cylinder_qposadr] = 0.0
            self.data.qvel[self.cylinder_qposadr] = 0.0
        else:
            try:
                self.data.qpos[self.cylinder_qposadr:self.cylinder_qposadr+3] = np.array(self.start_cylinder)
                self.data.qvel[self.cylinder_qposadr:self.cylinder_qposadr+3] = 0.0
            except Exception as e:
                print(f"Warning in reset: cannot set cylinder qpos → {e}")

        mujoco.mj_forward(self.model, self.data)

        self.tray_pos = self.start_tray.copy()
        self.tray_yaw = self.start_yaw

        return self._get_obs()

    def step(self, action):
        self.t += 1

        # Tray control approximation
        dx, dy, dz, dyaw = action
        self.tray_pos += np.array([dx, dy, dz])
        self.tray_yaw += dyaw

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Rewards
        cylinder_state = self._get_cylinder_xyz()
        tray_dist = np.linalg.norm(self.tray_pos - self.goal_tray) + abs(self.tray_yaw - self.goal_yaw)
        cylinder_offset = np.linalg.norm(cylinder_state[:2] - self.tray_pos[:2])
        reward = -tray_dist - cylinder_offset

        done = self.t >= self.max_steps or cylinder_state[2] < 0.1 or tray_dist < 0.05
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
        cylinder_state = self._get_cylinder_xyz()
        rel_cylinder = cylinder_state - self.tray_pos
        return np.concatenate([rel_cylinder, self.tray_pos, [self.tray_yaw]])

    def render(self, mode="human"):
        cylinder_state = self._get_cylinder_xyz()
        print(f"Step {self.t}: Tray {self.tray_pos}, Cylinder {cylinder_state}")

    def close(self):
        pass