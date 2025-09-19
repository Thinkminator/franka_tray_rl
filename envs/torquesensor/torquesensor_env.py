import gym
import numpy as np
import mujoco
from gym import spaces


class TorqueSensorEnv(gym.Env):
    """
    Custom environment for tray-ball manipulation (torquesensor variant).
    Observation uses end-effector torque sensor instead of ball position.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, model_path="assets/panda_tray_torque/panda_tray_ball_torque.xml"):
        super(TorqueSensorEnv, self).__init__()

        # Load Mujoco model (robot + tray + ball)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Action: Δtray_x, Δtray_y, Δtray_z, Δyaw
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(4,), dtype=np.float32)

        # Observation: [τy, τz] + tray_pos(3) + tray_yaw(1)
        obs_dim = 2 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define start & goal tray configs
        self.start_tray = np.array([0.5, 0.3, 0.65])
        self.start_yaw = -3.07
        self.start_joints = np.array([-0.019, 1.424, -0.833, 1.304, 0.174, 1.803, -0.004])

        self.goal_tray = np.array([0.5, -0.35, 0.2])
        self.goal_yaw = -4.57

        self.t = 0
        self.max_steps = 500

        # Sensor ID cache
        self.torque_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque")
        self.torque_adr = self.model.sensor_adr[self.torque_id]
        self.torque_dim = self.model.sensor_dim[self.torque_id]

    def reset(self):
        self.t = 0

        # Reset robot joints
        for i, q in enumerate(self.start_joints):
            self.data.qpos[i] = q

        # Reset ball position
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        ball_qposadr = self.model.jnt_qposadr[ball_id]
        self.data.qpos[ball_qposadr:ball_qposadr+7] = np.array([0.5, 0.3, 0.7, 1, 0, 0, 0])

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

        # Rewards: reach goal tray pose (no direct ball state)
        tray_dist = np.linalg.norm(self.tray_pos - self.goal_tray) + abs(self.tray_yaw - self.goal_yaw)
        reward = -tray_dist

        done = self.t >= self.max_steps or tray_dist < 0.05
        return obs, reward, done, {}

    def _get_obs(self):
        # Read torque from sensor: outputs [τx, τy, τz]
        torques = self.data.sensordata[self.torque_adr:self.torque_adr + self.torque_dim]
        tau_y, tau_z = torques[1], torques[2]   # keep only Y, Z
        return np.concatenate([[tau_y, tau_z], self.tray_pos, [self.tray_yaw]])

    def render(self, mode="human"):
        torques = self.data.sensordata[self.torque_adr:self.torque_adr + self.torque_dim]
        print(f"Step {self.t}: Tray {self.tray_pos}, τy={torques[1]:.4f}, τz={torques[2]:.4f}")

    def close(self):
        pass