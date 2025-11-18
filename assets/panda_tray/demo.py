"""Franka Emika MuJoCo demo — effort joint position control with MuJoCo Playground viewer (if available)."""

import time
from threading import Thread

import mujoco
import numpy as np

# Optional import for fallback rendering
try:
    import glfw
    GLFW_AVAILABLE = True
except Exception:
    GLFW_AVAILABLE = False


class Demo:

    qpos0 = [0.63, 0.17, 0.84, -1.01, -0.55, 2.68, -0.43]
    height, width = 480, 640  # Rendering window resolution (used only for fallback).
    fps = 30  # Rendering framerate (fallback).

    def __init__(self, xml_path: str = "assets/panda_tray/world.xml") -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True

        # Joint/Actuator setup (precompute indices)
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.dof_addrs = [self.model.joint(name).dofadr for name in self.joint_names]
        self.act_ids = [self.model.actuator(name).id for name in self.joint_names]

        # Initialize joint positions
        for i, q in enumerate(self.qpos0):
            self.data.qpos[self.dof_addrs[i]] = float(q)
        mujoco.mj_forward(self.model, self.data)

        # Gains for effort joint position control (7 joints)
        self.Kp = np.array([600.0, 600.0, 600.0, 30.0, 30.0, 30.0, 30.0], dtype=np.float64)
        self.Kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        self.torque_rate_limit = 1.0  # Max torque change per control step (tunable)
        self.pos_d = np.array(self.qpos0, dtype=np.float64)  # Desired joint positions initialized to start pose
        self.prev_torque = np.zeros(7, dtype=np.float64)  # Previous torque command for rate limiting

    def saturate_torque_rate(self, tau_d_calculated, tau_prev):
        delta_tau = tau_d_calculated - tau_prev
        delta_tau = np.clip(delta_tau, -self.torque_rate_limit, self.torque_rate_limit)
        return tau_prev + delta_tau

    def control(self):
        # Read current joint positions and velocities using dof addresses (scalars)
        qpos = np.zeros(7, dtype=np.float64)
        qvel = np.zeros(7, dtype=np.float64)
        coriolis = np.zeros(7, dtype=np.float64)
        
        for i, dof_addr in enumerate(self.dof_addrs):
            qpos[i] = self.data.qpos[dof_addr].item()
            qvel[i] = self.data.qvel[dof_addr].item()
            coriolis[i] = self.data.qfrc_bias[dof_addr].item()

        # Compute position and velocity errors
        pos_error = self.pos_d - qpos
        vel_error = -qvel  # Desired velocity is zero

        # PD control torque command
        tau_d = coriolis + self.Kp * pos_error + self.Kd * vel_error

        # Torque rate saturation
        tau_d_saturated = self.saturate_torque_rate(tau_d, self.prev_torque)
        self.prev_torque = tau_d_saturated.copy()

        # Apply torque commands to actuators via data.ctrl using actuator indices
        for i, act_id in enumerate(self.act_ids):
            # Ensure a Python float is written into data.ctrl
            self.data.ctrl[act_id] = float(tau_d_saturated[i])

    def step(self) -> None:
        # Main physics loop in a separate thread
        while self.run:
            try:
                self.control()
                mujoco.mj_step(self.model, self.data)
            except Exception:
                # Stop gracefully on unexpected error
                self.run = False
                raise
            # small sleep to avoid busy looping; physics timestep is governed by model.opt.timestep
            time.sleep(1e-3)

    def render(self) -> None:
        # Try to use MuJoCo's built-in interactive viewer (Playground)
        try:
            viewer = getattr(mujoco, "viewer", None)
            if viewer is not None and hasattr(viewer, "launch"):
                # This will block until the viewer window is closed.
                viewer.launch(self.model, self.data)
                # When the viewer closes, stop physics loop
                self.run = False
                return
        except Exception:
            # If anything fails here, fall back to manual GLFW rendering below
            pass

        # Fallback: GLFW + MJR rendering (basic view). Requires glfw to be available.
        if not GLFW_AVAILABLE:
            raise RuntimeError("Neither mujoco.viewer.launch available nor glfw installed for fallback rendering.")

        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Demo (fallback)", None, None)
        glfw.make_context_current(window)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        cam = mujoco.MjvCamera()
        # Use free camera mode so user can manipulate it with mouse
        try:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        except Exception:
            pass

        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()

        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.step, daemon=True)
        step_thread.start()
        self.render()


if __name__ == "__main__":
    Demo().start()