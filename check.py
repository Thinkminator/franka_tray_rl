import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("assets/panda_tray/panda_tray_cylinder_torque.xml")
data = mujoco.MjData(model)

# Compute forward kinematics
mujoco.mj_forward(model, data)

# Get site IDs
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tray_base")

# World poses
pos_ee = data.site_xpos[ee_id]
mat_ee = data.site_xmat[ee_id].reshape(3, 3)

pos_tray = data.site_xpos[tray_id]
mat_tray = data.site_xmat[tray_id].reshape(3, 3)

print("EE site world pos:", pos_ee)
print("EE site rotation:\n", mat_ee)
print("\nTray site world pos:", pos_tray)
print("Tray site rotation:\n", mat_tray)

# -----------------------------
# Compute relative transform (EE -> Tray)
# -----------------------------
R_rel = mat_ee.T @ mat_tray        # relative rotation
t_rel = mat_ee.T @ (pos_tray - pos_ee)   # relative translation

print("\nRelative rotation (EE→Tray):\n", R_rel)
print("Relative translation (EE→Tray):", t_rel)

# Quick check: tray x-axis in EE frame
tray_x_in_world = mat_tray[:, 0]          # x-axis of tray expressed in world
tray_x_in_ee = mat_ee.T @ tray_x_in_world # express in EE frame basis
print("\nTray X-axis in EE frame:", tray_x_in_ee)