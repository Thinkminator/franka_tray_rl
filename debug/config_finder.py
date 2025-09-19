import pybullet as p
import pybullet_data
import numpy as np
import os
import csv

# -------------------------------
# PROJECT PATH SETUP
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

URDF_PATH = "assets/panda_tray/panda_tray.urdf"
CSV_PATH  = "jointpos/tray_feasible_configs.csv"

# -------------------------------
# SETUP PYBULLET
# -------------------------------
p.connect(p.DIRECT)   # headless
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF(URDF_PATH, useFixedBase=True)

# Identify tray_base link index
num_joints = p.getNumJoints(robot)
tray_link_index = -1
for i in range(num_joints):
    if p.getJointInfo(robot, i)[12].decode("utf-8") == "tray_base":
        tray_link_index = i
        print("Found tray_base at index:", tray_link_index)
        break

# -------------------------------
# SAMPLING SPACE
# -------------------------------
# Define workspace grid (adjust resolution as needed!)
x_range = np.linspace(0.3, 0.7, 5)   # forward/back
y_range = np.linspace(-0.3, 0.3, 5)  # left/right
z_range = np.linspace(0.1, 0.5, 3)   # height
yaw_range = np.linspace(-np.pi, np.pi, 12)  # tray flat orientation around z

# -------------------------------
# CSV OUTPUT
# -------------------------------
header = ["tray_x","tray_y","tray_z","tray_yaw"] + [f"joint{i+1}" for i in range(7)]
rows = []

# -------------------------------
#  LOOP THROUGH POSE CANDIDATES
# -------------------------------
for x in x_range:
    for y in y_range:
        for z in z_range:
            for yaw in yaw_range:

                target_pos = [x, y, z]
                target_orn = p.getQuaternionFromEuler([0, 0, yaw])  # tray flat

                # Run IK
                joint_positions = p.calculateInverseKinematics(
                    bodyUniqueId=robot,
                    endEffectorLinkIndex=tray_link_index,
                    targetPosition=target_pos,
                    targetOrientation=target_orn,
                    maxNumIterations=200,
                    residualThreshold=1e-4
                )

                # Apply joints temporarily for collision check
                for j in range(7):  # Panda DOFs
                    p.resetJointState(robot, j, joint_positions[j])

                p.stepSimulation()

                # Collision checking
                contacts = p.getContactPoints(bodyA=robot)
                if len(contacts) > 0:
                    # In collision, skip
                    continue

                # Record feasible solution
                row = [x, y, z, yaw] + list(joint_positions[:7])
                rows.append(row)

print(f"Found {len(rows)} feasible poses")

# -------------------------------
# SAVE TO CSV
# -------------------------------
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Saved feasible configs to {CSV_PATH}")

p.disconnect()