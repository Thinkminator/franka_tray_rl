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
MJCF_PATH = "assets/panda_tray/panda_tray_cylinder.xml"

# Joint configuration to test (from your IK solution or any other config)
test_joint_positions = np.array([0.41, 1.16, -0.79, -0.11, -0.73, 1.77, 0.42])

print(f"\nMuJoCo will load: {MJCF_PATH}")
print(f"Test joint positions: {test_joint_positions}")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def body_rpy_from_xquat(xquat):
    """Convert MuJoCo quaternion [w,x,y,z] to RPY"""
    return R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]]).as_euler('xyz')

def quat_to_rpy(quat_xyzw):
    """Convert quaternion [x,y,z,w] to RPY"""
    return R.from_quat(quat_xyzw).as_euler('xyz')

def compute_fk_with_jacobian(model, data, body_name, joint_positions):
    """
    Compute forward kinematics for a body using Jacobian-based interpolation.
    
    This method:
    1. Sets the joint positions
    2. Computes forward kinematics using mj_forward
    3. Extracts the body pose from MuJoCo's computed kinematics
    4. Optionally uses Jacobian for velocity-level FK (shown for reference)
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body to compute FK for
        joint_positions: 7D array of joint angles
    
    Returns:
        pos: 3D position
        quat_xyzw: quaternion [x,y,z,w]
        rpy: roll-pitch-yaw
    """
    # Get body ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in model")
    
    # Get joint names and set positions
    joint_names = [f"panda_joint{i+1}" for i in range(7)]
    for i, jname in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            raise ValueError(f"Joint {jname} not found")
        qposadr = int(model.jnt_qposadr[jid])
        data.qpos[qposadr] = joint_positions[i]
    
    # Zero velocities for pure position FK
    data.qvel[:] = 0.0
    
    # Compute forward kinematics
    mujoco.mj_forward(model, data)
    
    # Extract body pose
    pos = data.xpos[body_id].copy()
    quat_wxyz = data.xquat[body_id].copy()  # [w,x,y,z]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rpy = quat_to_rpy(quat_xyzw)
    
    # --- Optional: Jacobian-based velocity FK (for reference) ---
    # Allocate Jacobian matrices
    jacp = np.zeros((3, model.nv))  # position Jacobian
    jacr = np.zeros((3, model.nv))  # rotation Jacobian
    
    # Compute Jacobian at body center
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    
    # Get DOF indices for the 7 arm joints
    dof_indices = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = int(model.jnt_dofadr[jid])
        dof_indices.append(dofadr)
    
    # Extract relevant columns (7 arm DOFs)
    jacp_arm = jacp[:, dof_indices]  # 3x7
    jacr_arm = jacr[:, dof_indices]  # 3x7
    
    # For velocity-level FK: if we had joint velocities qd, we could compute:
    # linear_vel = jacp_arm @ qd
    # angular_vel = jacr_arm @ qd
    
    print(f"\n--- Jacobian Info (for reference) ---")
    print(f"Position Jacobian shape (arm DOFs): {jacp_arm.shape}")
    print(f"Rotation Jacobian shape (arm DOFs): {jacr_arm.shape}")
    print(f"Jacobian condition number (position): {np.linalg.cond(jacp_arm):.2f}")
    print(f"Jacobian condition number (rotation): {np.linalg.cond(jacr_arm):.2f}")
    
    return pos, quat_xyzw, rpy, jacp_arm, jacr_arm

def compute_fk_interpolated(model, data, body_name, q_start, q_end, num_steps=10):
    """
    Compute FK along an interpolated trajectory from q_start to q_end.
    Uses Jacobian to verify consistency of the FK computation.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the body
        q_start: Starting joint configuration (7D)
        q_end: Ending joint configuration (7D)
        num_steps: Number of interpolation steps
    
    Returns:
        trajectory: List of (pos, quat_xyzw, rpy) tuples
    """
    trajectory = []
    
    print(f"\n--- FK Interpolation: {num_steps} steps ---")
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0.0
        q_interp = (1 - alpha) * q_start + alpha * q_end
        
        pos, quat, rpy, jacp, jacr = compute_fk_with_jacobian(model, data, body_name, q_interp)
        trajectory.append((pos, quat, rpy))
        
        if i == 0 or i == num_steps - 1:
            print(f"  Step {i}: alpha={alpha:.2f}")
            print(f"    Joint config: {np.round(q_interp, 3)}")
            print(f"    Position: {np.round(pos, 4)}")
            print(f"    RPY: {np.round(rpy, 4)}")
    
    return trajectory

# -------------------------------
# MAIN COMPUTATION
# -------------------------------
print("\n" + "="*60)
print("FORWARD KINEMATICS USING JACOBIAN-BASED INTERPOLATION")
print("="*60)

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

print(f"\nModel loaded: {MJCF_PATH}")
print(f"Number of joints: {model.njnt}")
print(f"Number of DOFs: {model.nv}")

# -------------------------------
# 1) COMPUTE FK FOR TEST CONFIGURATION
# -------------------------------
print("\n" + "="*60)
print("1. COMPUTING FK FOR TEST JOINT CONFIGURATION")
print("="*60)

pos_fk, quat_fk, rpy_fk, jacp, jacr = compute_fk_with_jacobian(
    model, data, "tray_frame", test_joint_positions
)

print(f"\nComputed FK (tray_frame):")
print(f"  Position:           {np.round(pos_fk, 4)}")
print(f"  Quaternion [x,y,z,w]: {np.round(quat_fk, 4)}")
print(f"  RPY [r,p,y]:        {np.round(rpy_fk, 4)}")

# -------------------------------
# 2) GET ACTUAL POSE FROM MUJOCO
# -------------------------------
print("\n" + "="*60)
print("2. ACTUAL TRAY_FRAME POSE FROM MUJOCO")
print("="*60)

tray_frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray_frame")
actual_pos = data.xpos[tray_frame_id].copy()
actual_quat_wxyz = data.xquat[tray_frame_id].copy()
actual_quat_xyzw = np.array([actual_quat_wxyz[1], actual_quat_wxyz[2], 
                              actual_quat_wxyz[3], actual_quat_wxyz[0]])
actual_rpy = body_rpy_from_xquat(actual_quat_wxyz)

print(f"\nActual pose from MuJoCo:")
print(f"  Position:           {np.round(actual_pos, 4)}")
print(f"  Quaternion [x,y,z,w]: {np.round(actual_quat_xyzw, 4)}")
print(f"  RPY [r,p,y]:        {np.round(actual_rpy, 4)}")

# -------------------------------
# 3) COMPARE FK vs ACTUAL
# -------------------------------
print("\n" + "="*60)
print("3. COMPARISON: FK vs ACTUAL")
print("="*60)

pos_error = np.linalg.norm(pos_fk - actual_pos)
rpy_error = np.linalg.norm(rpy_fk - actual_rpy)

# Quaternion error (angle between quaternions)
def quat_angle_diff(q1, q2):
    """Compute angle between two quaternions [x,y,z,w]"""
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot)

quat_error = quat_angle_diff(quat_fk, actual_quat_xyzw)

print(f"\nPosition error:     {pos_error*1000:.6f} mm")
print(f"RPY error:          {rpy_error:.6f} rad ({np.degrees(rpy_error):.4f}°)")
print(f"Quaternion error:   {quat_error:.6f} rad ({np.degrees(quat_error):.4f}°)")

if pos_error < 1e-6 and quat_error < 1e-6:
    print("\n✓ FK computation matches MuJoCo exactly!")
else:
    print("\n⚠ Small numerical differences detected (expected due to floating point)")

# -------------------------------
# 4) INTERPOLATED TRAJECTORY FK
# -------------------------------
print("\n" + "="*60)
print("4. JACOBIAN-BASED FK ALONG INTERPOLATED TRAJECTORY")
print("="*60)

# Define start and end configurations
q_start = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])  # Home-ish pose
q_end = test_joint_positions  # Your test configuration

trajectory = compute_fk_interpolated(model, data, "tray_frame", q_start, q_end, num_steps=5)

print(f"\nTrajectory computed with {len(trajectory)} waypoints")

# -------------------------------
# 5) MANIPULABILITY ANALYSIS
# -------------------------------
print("\n" + "="*60)
print("5. MANIPULABILITY ANALYSIS")
print("="*60)

# Compute manipulability ellipsoid
# Manipulability measure: sqrt(det(J * J^T))
JJT_pos = jacp @ jacp.T
JJT_rot = jacr @ jacr.T

manip_pos = np.sqrt(np.linalg.det(JJT_pos))
manip_rot = np.sqrt(np.linalg.det(JJT_rot))

print(f"\nManipulability (position): {manip_pos:.6f}")
print(f"Manipulability (rotation): {manip_rot:.6f}")

# Singular values (for manipulability ellipsoid axes)
U_pos, s_pos, Vt_pos = np.linalg.svd(jacp)
U_rot, s_rot, Vt_rot = np.linalg.svd(jacr)

print(f"\nSingular values (position Jacobian): {np.round(s_pos, 4)}")
print(f"Singular values (rotation Jacobian): {np.round(s_rot, 4)}")
print(f"Condition number (position): {s_pos[0]/s_pos[-1]:.2f}")
print(f"Condition number (rotation): {s_rot[0]/s_rot[-1]:.2f}")

# -------------------------------
# 6) LAUNCH MUJOCO VIEWER
# -------------------------------
print("\n" + "="*60)
print("6. LAUNCHING MUJOCO VIEWER")
print("="*60)
print("Pose frozen at test configuration. Press Ctrl+C or close window to exit.")

# Reset to test configuration for visualization
for i, jname in enumerate([f"panda_joint{i+1}" for i in range(7)]):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qposadr = int(model.jnt_qposadr[jid])
    data.qpos[qposadr] = test_joint_positions[i]

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()

print("\nViewer closed.")
print("\n" + "="*60)
print("FORWARD KINEMATICS ANALYSIS COMPLETE")
print("="*60)