import tkinter as tk
import numpy as np
import pybullet as p
import pybullet_data
import mujoco
import mujoco.viewer
import os
from multiprocessing import Process, Queue
from scipy.spatial.transform import Rotation as R

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

URDF_PATH = "assets/panda_tray/panda_tray.urdf"
MJCF_PATH = "assets/panda_tray/panda_tray_cylinder.xml"

# Panda joint limits (rad)
JOINT_LIMITS_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JOINT_LIMITS_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
JOINT_REST_POSE = [0, 0, 0, -1.0, 0, 1.5, 0]

# Workspace limits for tray_frame
WORKSPACE_LIMITS = {
    'x': (0.2, 0.8),
    'y': (-0.5, 0.5),
    'z': (0.2, 1.0)
}

# -------------------------------
# UI PROCESS
# -------------------------------
def start_ui_process(command_queue, status_queue):
    """Tkinter UI running in separate process"""
    
    root = tk.Tk()
    root.title("Tray IK Controller")
    
    # State variables (updated from MuJoCo)
    current_pose = [float('nan')] * 4  # [x, y, z, yaw]
    current_joints = [0.0] * 7
    
    # Tray control buttons
    tk.Label(root, text="Tray Controls", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=4, pady=5)
    
    controls = [
        ("X+", 0, 0.05), ("X-", 0, -0.05),
        ("Y+", 1, 0.05), ("Y-", 1, -0.05),
        ("Z+", 2, 0.05), ("Z-", 2, -0.05),
        ("Yaw+", 3, 0.1), ("Yaw-", 3, -0.1)
    ]
    
    for i, (label, axis, delta) in enumerate(controls):
        row = 1 + i // 2
        col = i % 2
        tk.Button(root, text=label, width=6,
                  command=lambda a=axis, d=delta: command_queue.put(("tray", (a, d)))
                  ).grid(row=row, column=col, padx=2, pady=2)
    
    # Current pose display
    tk.Label(root, text="Current Tray Pose (tray_frame)", 
             font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=4, pady=(10,2))
    pose_label = tk.Label(root, text="Waiting for data...", font=("Arial", 9))
    pose_label.grid(row=6, column=0, columnspan=4)
    
    # Joint sliders
    tk.Label(root, text="Joint Controls", font=("Arial", 12, "bold")).grid(
        row=7, column=0, columnspan=4, pady=(10,5))
    
    joint_sliders = []
    for i in range(7):
        tk.Label(root, text=f"Joint {i+1}").grid(row=8+i, column=0, sticky="w", padx=5)
        slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01,
                          orient=tk.HORIZONTAL, length=250)
        slider.set(0.0)
        slider.grid(row=8+i, column=1, columnspan=2, padx=5, pady=2)
        joint_sliders.append(slider)
    
    tk.Button(root, text="Set Joints", bg="lightblue", font=("Arial", 10, "bold"),
              command=lambda: command_queue.put(("joints", [s.get() for s in joint_sliders]))
              ).grid(row=15, column=0, columnspan=3, pady=10)
    
    # Save configuration button
    tk.Button(root, text="💾 Save Config", bg="lightgreen", font=("Arial", 10, "bold"),
              command=lambda: command_queue.put(("save", None))
              ).grid(row=16, column=0, columnspan=3, pady=5)
    
    def update_display():
        """Update display with latest status from MuJoCo"""
        try:
            while not status_queue.empty():
                status_type, data = status_queue.get_nowait()
                if status_type == "pose":
                    current_pose[:] = data
                elif status_type == "joints":
                    current_joints[:] = data
                    for i, slider in enumerate(joint_sliders):
                        slider.set(current_joints[i])
        except:
            pass
        
        # Update pose display
        if not np.isnan(current_pose[0]):
            pose_text = f"X:{current_pose[0]:.3f} Y:{current_pose[1]:.3f} Z:{current_pose[2]:.3f} Yaw:{current_pose[3]:.3f}"
        else:
            pose_text = "Waiting for data..."
        pose_label.config(text=pose_text)
        
        root.after(100, update_display)
    
    update_display()
    root.mainloop()

# -------------------------------
# MUJOCO PROCESS
# -------------------------------
def run_mujoco_with_ui(command_queue, status_queue):
    """MuJoCo simulation with UI command processing"""
    
    # Setup PyBullet (for IK)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    # Find tray_frame link index
    tray_link_index = -1
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode("utf-8") == "tray_frame":
            tray_link_index = i
            break
    if tray_link_index == -1:
        raise RuntimeError("tray_frame not found in PyBullet URDF")
    print(f"Found tray_frame in PyBullet at index: {tray_link_index}")
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    # Verify tray_frame exists in MuJoCo
    tray_frame_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray_frame")
    if tray_frame_id == -1:
        raise RuntimeError("tray_frame body not found in MuJoCo MJCF")
    print(f"Found tray_frame in MuJoCo at body id: {tray_frame_id}")
    
    # Joint name mapping
    pb_joint_names = [f"panda_joint{i+1}" for i in range(7)]
    
    # State variables
    tray_pos = [0.65, 1.35, 0.8]  # Reachable starting position
    tray_yaw = -1.47
    joint_positions = [0.0] * 7
    
    def get_tray_pose_from_mujoco():
        """Get actual tray_frame pose from MuJoCo"""
        pos = data.xpos[tray_frame_id].copy()
        quat = data.xquat[tray_frame_id].copy()  # [w,x,y,z]
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        return [float(pos[0]), float(pos[1]), float(pos[2]), float(rpy[2])]
    
    def clamp_target():
        """Clamp target to reachable workspace"""
        tray_pos[0] = float(np.clip(tray_pos[0], *WORKSPACE_LIMITS['x']))
        tray_pos[1] = float(np.clip(tray_pos[1], *WORKSPACE_LIMITS['y']))
        tray_pos[2] = float(np.clip(tray_pos[2], *WORKSPACE_LIMITS['z']))
        nonlocal tray_yaw
        tray_yaw = float((tray_yaw + np.pi) % (2*np.pi) - np.pi)
    
    def apply_pose():
        """Run IK and update MuJoCo"""
        nonlocal joint_positions
        
        clamp_target()
        target_orn = p.getQuaternionFromEuler([0, 0, tray_yaw])
        
        # Calculate IK with joint limits
        joint_ranges = [u - l for l, u in zip(JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)]
        joint_positions = p.calculateInverseKinematics(
            robot, tray_link_index, tray_pos, target_orn,
            lowerLimits=JOINT_LIMITS_LOWER,
            upperLimits=JOINT_LIMITS_UPPER,
            jointRanges=joint_ranges,
            restPoses=JOINT_REST_POSE,
            maxNumIterations=200,
            residualThreshold=1e-4
        )[:7]
        
        # Apply to MuJoCo
        for i, name in enumerate(pb_joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = joint_positions[i]
        
        mujoco.mj_forward(model, data)
        
        # Get achieved pose and send to UI
        achieved_pose = get_tray_pose_from_mujoco()
        status_queue.put(("pose", achieved_pose))
        status_queue.put(("joints", list(joint_positions)))
        
        # Log error
        err = np.array(tray_pos + [tray_yaw]) - np.array(achieved_pose)
        pos_err = np.linalg.norm(err[:3])
        yaw_err = abs((err[3] + np.pi) % (2*np.pi) - np.pi)
        
        print(f"Target: {np.round(tray_pos,3).tolist()} Yaw:{tray_yaw:.2f} | "
              f"Achieved: {np.round(achieved_pose,3).tolist()} | "
              f"Err: {pos_err*1000:.1f}mm, {np.degrees(yaw_err):.1f}° | "
              f"Joints: {[round(j,2) for j in joint_positions]}")
    
    def apply_joints(joint_values):
        """Apply joint values directly"""
        nonlocal joint_positions
        joint_positions = joint_values[:7]
        
        # Apply to MuJoCo
        for i, name in enumerate(pb_joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                data.qpos[model.jnt_qposadr[jid]] = joint_positions[i]
        
        mujoco.mj_forward(model, data)
        
        # Get achieved pose and send to UI
        achieved_pose = get_tray_pose_from_mujoco()
        status_queue.put(("pose", achieved_pose))
        status_queue.put(("joints", list(joint_positions)))
        
        print(f"Joints: {[round(j,2) for j in joint_positions]} | "
              f"Achieved: {np.round(achieved_pose,3).tolist()}")
    
    def save_config():
        """Save current configuration to CSV"""
        import csv
        achieved_pose = get_tray_pose_from_mujoco()
        
        filename = "current_config.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["tray_x", "tray_y", "tray_z", "tray_yaw"] + 
                            [f"joint{i+1}" for i in range(7)])
            writer.writerow(achieved_pose + list(joint_positions))
        
        print(f"✓ Configuration saved to {filename}")
        print(f"  Pose: {np.round(achieved_pose,3).tolist()}")
        print(f"  Joints: {[round(j,3) for j in joint_positions]}")
    
    # Initialize
    print("Setting initial pose...")
    apply_pose()
    
    # MuJoCo viewer loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer started. Use the UI window to control the robot.")
        
        while viewer.is_running():
            try:
                while not command_queue.empty():
                    cmd_type, value = command_queue.get_nowait()
                    
                    if cmd_type == "tray":
                        axis, delta = value
                        if axis < 3:
                            tray_pos[axis] += delta
                        else:
                            tray_yaw += delta
                        apply_pose()
                    
                    elif cmd_type == "joints":
                        apply_joints(value)
                    
                    elif cmd_type == "save":
                        save_config()
            except:
                pass
            
            viewer.sync()
    
    p.disconnect()

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    command_queue = Queue()
    status_queue = Queue()
    
    ui_process = Process(target=start_ui_process, args=(command_queue, status_queue))
    ui_process.start()
    
    try:
        run_mujoco_with_ui(command_queue, status_queue)
    finally:
        ui_process.terminate()
        ui_process.join()