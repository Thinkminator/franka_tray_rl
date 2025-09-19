import mujoco
import mujoco.viewer
import os
import sys

# Force working directory to project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Go 1 directory up if script is inside "scripts/"
PROJECT_ROOT = os.path.join(PROJECT_ROOT, "..")
os.chdir(PROJECT_ROOT)
print("Working directory set to:", os.getcwd())

# Load URDF directly with mujoco
mj_model = mujoco.MjModel.from_xml_path("assets/panda_tray/panda_tray.urdf")
data = mujoco.MjData(mj_model)

# Launch viewer
with mujoco.viewer.launch_passive(mj_model, data) as viewer:
    print("MuJoCo viewer opened!")
    print("Press 'Tab' → Visualize → check 'World frame' and 'Geom frames' to see coordinate axes")
    
    while viewer.is_running():
        mujoco.mj_step(mj_model, data)
        viewer.sync()