import pybullet as p
import pybullet_data
import numpy as np
import os
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# -------------------------------
# CONFIGURATION
# -------------------------------
URDF_PATH = "assets/panda_tray/panda_tray.urdf"
MJCF_PATH = "assets/panda_tray/panda_tray_cylinder.xml"

# Choose which model to load in MuJoCo: "urdf" or "xml"
MUJOCO_MODEL_TYPE = "xml"  # Change to "urdf" to load URDF instead

# Desired tray target pose
target_pos = [0.785, 0.107, 0.619]
target_rpy = [0, 0, 2.11]

print(f"\nMuJoCo will load: {MUJOCO_MODEL_TYPE.upper()}")
print(f"Target pose:")
print(f"  Position: {target_pos}")
print(f"  RPY: {target_rpy}")

# -------------------------------
# 1) GET IK FROM PYBULLET
# -------------------------------
p.connect(p.DIRECT)   # run headless (no GUI needed, we just want IK)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF(URDF_PATH, useFixedBase=True)

# Find tray_frame link index
num_joints = p.getNumJoints(robot)
tray_link_index = -1
for i in range(num_joints):
    link_name = p.getJointInfo(robot, i)[12].decode("utf-8")
    if link_name == "tray_frame":
        tray_link_index = i
        print(f"Found tray_frame at PyBullet index: {tray_link_index}")
        break

if tray_link_index == -1:
    raise RuntimeError("tray_frame link not found in PyBullet URDF")

# Convert RPY to quaternion
target_orn = p.getQuaternionFromEuler(target_rpy)

print("\nRunning IK in PyBullet...")
joint_positions = p.calculateInverseKinematics(
    bodyUniqueId=robot,
    endEffectorLinkIndex=tray_link_index,
    targetPosition=target_pos,
    targetOrientation=target_orn,
    maxNumIterations=200,
    residualThreshold=1e-4
)

print(f"IK solution (7 joints): {[round(j, 3) for j in joint_positions[:7]]}")
p.disconnect()   # close pybullet

# -------------------------------
# 2) LOAD IN MUJOCO
# -------------------------------
if MUJOCO_MODEL_TYPE == "urdf":
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    print(f"Loaded URDF in MuJoCo: {URDF_PATH}")
elif MUJOCO_MODEL_TYPE == "xml":
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    print(f"Loaded XML in MuJoCo: {MJCF_PATH}")
else:
    raise ValueError(f"Invalid MUJOCO_MODEL_TYPE: {MUJOCO_MODEL_TYPE}")

data = mujoco.MjData(model)

# Get MuJoCo joint names
mj_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]

print(f"\nMuJoCo joints: {mj_joint_names}")

# PyBullet / Panda joint naming convention (first 7 actuated joints)
pb_joint_names = [f"panda_joint{i+1}" for i in range(7)]

# Map IK solution into MuJoCo by joint name
for i, name in enumerate(pb_joint_names):
    if name in mj_joint_names:
        mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qposadr = int(model.jnt_qposadr[mj_id])
        data.qpos[qposadr] = joint_positions[i]

# Update kinematics
mujoco.mj_forward(model, data)

# -------------------------------
# 3) GET TRAY_FRAME POSE
# -------------------------------
def body_rpy_from_xquat(xquat):
    """Convert MuJoCo quaternion [w,x,y,z] to RPY"""
    # scipy expects [x,y,z,w]
    return R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]]).as_euler('xyz')

# Get tray_frame body ID
tray_frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray_frame")
if tray_frame_id == -1:
    print("\n⚠ Error: tray_frame not found in MuJoCo model")
else:
    # Get actual tray_frame pose in MuJoCo
    actual_pos = data.xpos[tray_frame_id].copy()
    actual_quat = data.xquat[tray_frame_id].copy()  # [w, x, y, z] format
    actual_rpy = body_rpy_from_xquat(actual_quat)
    
    print("\n" + "="*60)
    print("TRAY_FRAME POSE IN MUJOCO")
    print("="*60)
    print(f"  Position:           {np.round(actual_pos, 4).tolist()}")
    print(f"  Quaternion [w,x,y,z]: {np.round(actual_quat, 4).tolist()}")
    print(f"  Quaternion [x,y,z,w]: {np.round([actual_quat[1], actual_quat[2], actual_quat[3], actual_quat[0]], 4).tolist()}")
    print(f"  RPY [r,p,y]:        {np.round(actual_rpy, 4).tolist()}")
    
    # Compute error
    pos_error = np.linalg.norm(np.array(target_pos) - actual_pos)
    yaw_error = abs((target_rpy[2] - actual_rpy[2] + np.pi) % (2*np.pi) - np.pi)
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"  Target position:    {target_pos}")
    print(f"  Achieved position:  {np.round(actual_pos, 4).tolist()}")
    print(f"  Position error:     {pos_error*1000:.2f} mm")
    print(f"  Target yaw:         {target_rpy[2]:.4f} rad ({np.degrees(target_rpy[2]):.2f}°)")
    print(f"  Achieved yaw:       {actual_rpy[2]:.4f} rad ({np.degrees(actual_rpy[2]):.2f}°)")
    print(f"  Yaw error:          {yaw_error:.4f} rad ({np.degrees(yaw_error):.2f}°)")
    print("="*60)

# -------------------------------
# 4) LAUNCH MUJOCO VIEWER (freeze joints)
# -------------------------------
print("\nLaunching MuJoCo viewer...")
print("Pose frozen at IK result. Press Ctrl+C or close window to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # NO mj_step() -> physics is not executed, pose is frozen
        viewer.sync()

print("\nViewer closed.")