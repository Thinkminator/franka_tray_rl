import tkinter as tk
import numpy as np
import pybullet as p
import pybullet_data
import mujoco
import mujoco.viewer
import os
from multiprocessing import Process, Queue
import time

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

URDF_PATH = "assets/panda_tray/panda_tray.urdf"

# -------------------------------
# UI PROCESS (Tkinter in separate process)
# -------------------------------
def start_ui_process(command_queue, status_queue):
    """Tkinter UI running in separate process"""
    
    def send_command(cmd_type, value):
        command_queue.put((cmd_type, value))
    
    root = tk.Tk()
    root.title("Tray IK Controller")
    
    # Current pose display variables
    current_pose = [0.5, 0.3, 0.2, -0.87]
    current_joints = [0.0] * 7
    
    # Tray control buttons
    tk.Label(root, text="Tray Controls", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=4, pady=5)
    
    tk.Button(root, text="X+", command=lambda: send_command("tray", (0, 0.05)), width=6).grid(row=1, column=0, padx=2, pady=2)
    tk.Button(root, text="X-", command=lambda: send_command("tray", (0, -0.05)), width=6).grid(row=1, column=1, padx=2, pady=2)
    tk.Button(root, text="Y+", command=lambda: send_command("tray", (1, 0.05)), width=6).grid(row=2, column=0, padx=2, pady=2)
    tk.Button(root, text="Y-", command=lambda: send_command("tray", (1, -0.05)), width=6).grid(row=2, column=1, padx=2, pady=2)
    tk.Button(root, text="Z+", command=lambda: send_command("tray", (2, 0.05)), width=6).grid(row=3, column=0, padx=2, pady=2)
    tk.Button(root, text="Z-", command=lambda: send_command("tray", (2, -0.05)), width=6).grid(row=3, column=1, padx=2, pady=2)
    tk.Button(root, text="Yaw+", command=lambda: send_command("tray", (3, 0.1)), width=6).grid(row=4, column=0, padx=2, pady=2)
    tk.Button(root, text="Yaw-", command=lambda: send_command("tray", (3, -0.1)), width=6).grid(row=4, column=1, padx=2, pady=2)
    
    # Current pose display
    tk.Label(root, text="Current Tray Pose", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=4, pady=(10,2))
    pose_label = tk.Label(root, text="", font=("Arial", 9))
    pose_label.grid(row=6, column=0, columnspan=4)
    
    # Joint sliders
    tk.Label(root, text="Joint Controls", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=4, pady=(10,5))
    
    joint_sliders = []
    for i in range(7):
        tk.Label(root, text=f"Joint {i+1}").grid(row=8+i, column=0, sticky="w", padx=5)
        slider = tk.Scale(root, from_=-3.14, to=3.14, resolution=0.01,
                          orient=tk.HORIZONTAL, length=250)
        slider.set(0.0)  # Initial value
        slider.grid(row=8+i, column=1, columnspan=2, padx=5, pady=2)
        joint_sliders.append(slider)
    
    def set_from_sliders():
        joint_values = [slider.get() for slider in joint_sliders]
        send_command("joints", joint_values)
    
    tk.Button(root, text="Set Joints", command=set_from_sliders, 
              bg="lightblue", font=("Arial", 10, "bold")).grid(row=15, column=0, columnspan=3, pady=10)
    
    # Save configuration button
    def save_config():
        send_command("save", None)
    
    tk.Button(root, text="💾 Save Config", command=save_config,
              bg="lightgreen", font=("Arial", 10, "bold")).grid(row=16, column=0, columnspan=3, pady=5)
    
    def update_display():
        """Update display with latest status from MuJoCo process"""
        try:
            while not status_queue.empty():
                status_type, data = status_queue.get_nowait()
                if status_type == "pose":
                    current_pose[:] = data
                elif status_type == "joints":
                    current_joints[:] = data
                    # Update sliders
                    for i, slider in enumerate(joint_sliders):
                        slider.set(current_joints[i])
        except:
            pass
        
        # Update pose display
        pose_text = f"X:{current_pose[0]:.3f} Y:{current_pose[1]:.3f} Z:{current_pose[2]:.3f} Yaw:{current_pose[3]:.3f}"
        pose_label.config(text=pose_text)
        
        root.after(100, update_display)  # Update every 100ms
    
    update_display()
    root.mainloop()

# -------------------------------
# MUJOCO PROCESS (Main thread)
# -------------------------------
def run_mujoco_with_ui(command_queue, status_queue):
    """MuJoCo simulation with UI command processing"""
    
    # Setup PyBullet
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)
    
    tray_link_index = -1
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode("utf-8") == "tray_base":
            tray_link_index = i
            print("Found tray_base at index:", tray_link_index)
            break
    
    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)
    
    mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                      for i in range(model.njnt)]
    pb_joint_names = [f"panda_joint{i+1}" for i in range(7)]
    
    # Starting configuration
    tray_pos = [0.5, 0.3, 0.2]
    tray_yaw = -0.87
    joint_positions = [0.0] * 7
    
    def apply_pose():
        """Run IK and update MuJoCo"""
        nonlocal joint_positions
        target_orn = p.getQuaternionFromEuler([0, 0, tray_yaw])
        
        joint_positions = p.calculateInverseKinematics(
            robot,
            tray_link_index,
            tray_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-4
        )
        
        # Apply to MuJoCo
        for i, name in enumerate(pb_joint_names):
            if name in mj_joint_names:
                mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[mj_id] = joint_positions[i]
        
        mujoco.mj_forward(model, data)
        
        # Send status to UI
        status_queue.put(("pose", tray_pos + [tray_yaw]))
        status_queue.put(("joints", list(joint_positions[:7])))
        
        print("Updated pose -> Tray:", [round(x,3) for x in tray_pos], "Yaw:", round(tray_yaw,2),
              "Joints:", [round(j,3) for j in joint_positions[:7]])
    
    def apply_joints(joint_values):
        """Apply joint values directly"""
        nonlocal joint_positions
        joint_positions = joint_values
        
        for i, name in enumerate(pb_joint_names):
            if name in mj_joint_names:
                mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[mj_id] = joint_positions[i]
        
        mujoco.mj_forward(model, data)
        status_queue.put(("joints", list(joint_positions[:7])))
        print("Applied joints directly ->", [round(j,3) for j in joint_positions])
    
    def save_config():
        """Save current configuration to CSV"""
        import csv
        filename = "current_config.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["tray_x", "tray_y", "tray_z", "tray_yaw"] + [f"joint{i+1}" for i in range(7)])
            writer.writerow(tray_pos + [tray_yaw] + list(joint_positions[:7]))
        print(f"Configuration saved to {filename}")
    
    # Initialize starting pose
    print("Setting initial pose...")
    apply_pose()
    
    # MuJoCo viewer loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer started. Use the UI window to control the robot.")
        
        while viewer.is_running():
            # Process UI commands
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
    # Create communication queues
    command_queue = Queue()  # UI -> MuJoCo
    status_queue = Queue()   # MuJoCo -> UI
    
    # Start UI process
    ui_process = Process(target=start_ui_process, args=(command_queue, status_queue))
    ui_process.start()
    
    try:
        # Run MuJoCo in main thread (required for OpenGL)
        run_mujoco_with_ui(command_queue, status_queue)
    finally:
        # Clean up
        ui_process.terminate()
        ui_process.join()