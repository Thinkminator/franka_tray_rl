import numpy as np
import pybullet as p
import pybullet_data
import os
import mujoco
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# Paths
URDF_PATH = "assets/panda_tray/panda_tray.urdf"
MJCF_PATH = "assets/panda_tray/panda_tray_cylinder.xml"

def quat_to_euler_zyx(quat_wxyz):
    """Convert quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw]"""
    return R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_euler('xyz')

def compare_fk():
    """Compare forward kinematics between PyBullet (URDF) and MuJoCo (MJCF)"""
    
    print("="*70)
    print("FK COMPARISON TEST: PyBullet URDF vs MuJoCo MJCF")
    print("="*70)
    
    # ========== SETUP PYBULLET ==========
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    # Find tray_frame link index in PyBullet
    tray_link_index = -1
    for i in range(p.getNumJoints(robot)):
        link_name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if link_name == "tray_frame":
            tray_link_index = i
            print(f"\n[PyBullet] Found tray_frame at link index: {tray_link_index}")
            break
    
    if tray_link_index == -1:
        print("[PyBullet] ERROR: tray_frame not found!")
        p.disconnect()
        return
    
    # Get the 7 revolute joint indices
    pb_joint_indices = []
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE and "panda_joint" in joint_name:
            pb_joint_indices.append(i)
    
    print(f"[PyBullet] Found {len(pb_joint_indices)} revolute joints: {pb_joint_indices}")
    
    # ========== SETUP MUJOCO ==========
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    # Find tray_frame body in MuJoCo
    tray_frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray_frame")
    if tray_frame_id == -1:
        print("[MuJoCo] ERROR: tray_frame body not found!")
        p.disconnect()
        return
    print(f"[MuJoCo] Found tray_frame at body id: {tray_frame_id}")
    
    # Get joint names and IDs
    mj_joint_names = [f"panda_joint{i+1}" for i in range(7)]
    mj_joint_ids = []
    for name in mj_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_joint_ids.append(jid)
    print(f"[MuJoCo] Joint IDs: {mj_joint_ids}")
    
    # ========== TEST 1: ZERO-JOINT FK ==========
    print("\n" + "="*70)
    print("TEST 1: ZERO-JOINT FK (all joints = 0)")
    print("="*70)
    
    # PyBullet: reset all joints to 0
    for idx in pb_joint_indices:
        p.resetJointState(robot, idx, 0.0)
    
    # Read tray_frame pose from PyBullet
    link_state = p.getLinkState(robot, tray_link_index, computeForwardKinematics=True)
    pb_pos = link_state[4]  # world position
    pb_quat = link_state[5]  # world orientation [x,y,z,w]
    pb_euler = p.getEulerFromQuaternion(pb_quat)
    
    print(f"\n[PyBullet] tray_frame pose (joints=0):")
    print(f"  Position: [{pb_pos[0]:.6f}, {pb_pos[1]:.6f}, {pb_pos[2]:.6f}]")
    print(f"  Quaternion [x,y,z,w]: [{pb_quat[0]:.6f}, {pb_quat[1]:.6f}, {pb_quat[2]:.6f}, {pb_quat[3]:.6f}]")
    print(f"  Euler [r,p,y]: [{pb_euler[0]:.6f}, {pb_euler[1]:.6f}, {pb_euler[2]:.6f}]")
    
    # MuJoCo: set all joints to 0
    for jid in mj_joint_ids:
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = 0.0
    
    mujoco.mj_forward(model, data)
    
    # Read tray_frame pose from MuJoCo
    mj_pos = data.xpos[tray_frame_id].copy()
    mj_quat = data.xquat[tray_frame_id].copy()  # [w,x,y,z]
    mj_euler = quat_to_euler_zyx(mj_quat)
    
    print(f"\n[MuJoCo] tray_frame pose (joints=0):")
    print(f"  Position: [{mj_pos[0]:.6f}, {mj_pos[1]:.6f}, {mj_pos[2]:.6f}]")
    print(f"  Quaternion [w,x,y,z]: [{mj_quat[0]:.6f}, {mj_quat[1]:.6f}, {mj_quat[2]:.6f}, {mj_quat[3]:.6f}]")
    print(f"  Euler [r,p,y]: [{mj_euler[0]:.6f}, {mj_euler[1]:.6f}, {mj_euler[2]:.6f}]")
    
    # Compute difference
    pos_diff = np.linalg.norm(np.array(pb_pos) - np.array(mj_pos))
    print(f"\n[DIFFERENCE] Position error: {pos_diff:.6f} m ({pos_diff*1000:.3f} mm)")
    
    # ========== TEST 2: IK-POSE FK ==========
    print("\n" + "="*70)
    print("TEST 2: IK-POSE FK (target pose with IK)")
    print("="*70)
    
    # Target pose for IK
    target_pos = [0.5, 0.0, 0.5]
    target_yaw = 0.0
    target_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
    
    print(f"\n[IK Target]")
    print(f"  Position: {target_pos}")
    print(f"  Yaw: {target_yaw}")
    
    # PyBullet IK
    joint_positions = p.calculateInverseKinematics(
        robot,
        tray_link_index,
        target_pos,
        target_orn,
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    
    print(f"\n[PyBullet] IK solution (7 joints):")
    print(f"  {[round(j, 4) for j in joint_positions[:7]]}")
    
    # Apply IK solution to PyBullet
    for i, idx in enumerate(pb_joint_indices):
        p.resetJointState(robot, idx, joint_positions[i])
    
    # Read tray_frame pose from PyBullet after IK
    link_state = p.getLinkState(robot, tray_link_index, computeForwardKinematics=True)
    pb_pos_ik = link_state[4]
    pb_quat_ik = link_state[5]
    pb_euler_ik = p.getEulerFromQuaternion(pb_quat_ik)
    
    print(f"\n[PyBullet] tray_frame pose after IK:")
    print(f"  Position: [{pb_pos_ik[0]:.6f}, {pb_pos_ik[1]:.6f}, {pb_pos_ik[2]:.6f}]")
    print(f"  Quaternion [x,y,z,w]: [{pb_quat_ik[0]:.6f}, {pb_quat_ik[1]:.6f}, {pb_quat_ik[2]:.6f}, {pb_quat_ik[3]:.6f}]")
    print(f"  Euler [r,p,y]: [{pb_euler_ik[0]:.6f}, {pb_euler_ik[1]:.6f}, {pb_euler_ik[2]:.6f}]")
    
    # Apply same joint angles to MuJoCo
    for i, jid in enumerate(mj_joint_ids):
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = joint_positions[i]
    
    mujoco.mj_forward(model, data)
    
    # Read tray_frame pose from MuJoCo
    mj_pos_ik = data.xpos[tray_frame_id].copy()
    mj_quat_ik = data.xquat[tray_frame_id].copy()
    mj_euler_ik = quat_to_euler_zyx(mj_quat_ik)
    
    print(f"\n[MuJoCo] tray_frame pose with same joints:")
    print(f"  Position: [{mj_pos_ik[0]:.6f}, {mj_pos_ik[1]:.6f}, {mj_pos_ik[2]:.6f}]")
    print(f"  Quaternion [w,x,y,z]: [{mj_quat_ik[0]:.6f}, {mj_quat_ik[1]:.6f}, {mj_quat_ik[2]:.6f}, {mj_quat_ik[3]:.6f}]")
    print(f"  Euler [r,p,y]: [{mj_euler_ik[0]:.6f}, {mj_euler_ik[1]:.6f}, {mj_euler_ik[2]:.6f}]")
    
    # Compute difference
    pos_diff_ik = np.linalg.norm(np.array(pb_pos_ik) - np.array(mj_pos_ik))
    print(f"\n[DIFFERENCE] Position error: {pos_diff_ik:.6f} m ({pos_diff_ik*1000:.3f} mm)")
    
    # ========== CLEANUP ==========
    p.disconnect()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nExpected result: Position errors should be < 1mm if models match.")
    print("If errors are large, check wrist transforms (EndEffector quat) in MJCF.\n")

if __name__ == "__main__":
    compare_fk()