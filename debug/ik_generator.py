import pybullet as p
import pybullet_data
import numpy as np
import os
import mujoco
import mujoco.viewer

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

URDF_PATH = "assets/panda_tray/panda_tray.urdf"

# -------------------------------
# 1) GET IK FROM PYBULLET
# -------------------------------
p.connect(p.DIRECT)   # run headless (no GUI needed, we just want IK)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF(URDF_PATH, useFixedBase=True)

# Find tray base link index
num_joints = p.getNumJoints(robot)
tray_link_index = -1
for i in range(num_joints):
    if p.getJointInfo(robot, i)[12].decode("utf-8") == "tray_base":
        tray_link_index = i
        print("Found tray_base at index:", tray_link_index)
        break

# Desired tray target pose
target_pos = [0.5, 0.3, 0.2]
target_orn = p.getQuaternionFromEuler([0, 0, -0.87])

joint_positions = p.calculateInverseKinematics(
    bodyUniqueId=robot,
    endEffectorLinkIndex=tray_link_index,
    targetPosition=target_pos,
    targetOrientation=target_orn,
    maxNumIterations=200,
    residualThreshold=1e-4
)

print("IK solution = ", joint_positions[:7])
p.disconnect()   # close pybullet

# -------------------------------
# 2) LOAD IN MUJOCO
# -------------------------------
model = mujoco.MjModel.from_xml_path(URDF_PATH)
data = mujoco.MjData(model)

# Get MuJoCo joint names
mj_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]

print("MuJoCo joints:", mj_joint_names)

# PyBullet / Panda joint naming convention (first 7 actuated joints)
pb_joint_names = [f"panda_joint{i+1}" for i in range(7)]

# Map IK solution into MuJoCo by joint name
for i, name in enumerate(pb_joint_names):
    if name in mj_joint_names:
        mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[mj_id] = joint_positions[i]

# Update kinematics
mujoco.mj_forward(model, data)

# -------------------------------
# 3) LAUNCH MUJOCO VIEWER (freeze joints)
# -------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Pose frozen at IK result. Press Ctrl+C to exit MuJoCo viewer")
    while viewer.is_running():
        # NO mj_step() -> physics is not executed, pose is frozen
        viewer.sync()