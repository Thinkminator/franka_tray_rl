import mujoco
import mujoco.viewer
import time
import os

def main():
    # Path to your XML file
    # Force working directory to project root
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    # Go 1 directory up if script is inside "scripts/"
    PROJECT_ROOT = os.path.join(PROJECT_ROOT, "..")
    os.chdir(PROJECT_ROOT)
    xml_path = "assets/panda_tray/panda_tray_ball.xml"
    
    # Check if file exists
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found!")
        return
    
    # Load the model directly
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set initial joint positions (optional - to see robot in a nice pose)
    # These correspond to your start_joints from the env
    start_joints = [-0.019, 1.424, -0.833, 1.304, 0.174, 1.803, -0.004]
    for i, q in enumerate(start_joints):
        if i < len(data.qpos):
            data.qpos[i] = q
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    
    print("Model loaded successfully!")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of geoms: {model.ngeom}")
    
    # List all bodies
    print("\nBodies in the model:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {i}: {body_name}")
    
    # Launch viewer
    print("\nLaunching MuJoCo viewer...")
    print("Press ESC to exit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Just sync the viewer (no physics stepping)
            viewer.sync()
            time.sleep(0.01)  # Small delay to prevent high CPU usage

if __name__ == "__main__":
    main()