🥼 Franka Tray RL


This repository contains a custom MuJoCo + Gym environment for robotic manipulation with a Franka Panda arm, where the robot holds a tray and must balance/manipulate a cylinder for reinforcement learning research.

    📂 Repository Structure
    franka_tray_rl/
    │
    ├── assets/                     # Models & meshes
    │   └── panda_tray/             # Panda + tray assets
    │       ├── hand.stl
    │       ├── link0.stl ... link7.stl   # Panda STL meshes
    │       ├── panda_tray.urdf     # URDF definition of Panda + Tray
    │       ├── panda_tray_cylinder.xml # MJCF (MuJoCo) with tray + cylinder
    |       ├── panda_tray_cylinder_camera.xml
    |       └── panda_tray_cylinder_torque.xml
    │
    ├── debug/                      # Standalone debugging scripts
    │   ├── config_finder.py        # Finds valid IK configurations
    │   ├── ik_generator.py         # Generates inverse kinematics
    │   ├── interactive_ik.py       # Interactive IK exploration
    │   ├── urdf_viewer.py          # Loads URDF in viewer
    │   └── xml_viewer.py           # Loads XML in MuJoCo viewer
    │
    ├── envs/                       # OpenAI Gym environments
    │   ├── depthcamera/            # (Placeholder for vision-based envs)
    │   ├── liquid/                 # (Placeholder for liquid simulation envs)
    │   ├── multi_object/             # (Future: tray with >1 cylinder)
    │   │   └── torquesensor_env.py
    │   └── traypose/               # Main tray manipulation environment
    │       ├── __init__.py
    │       ├── config.yaml         # Config file (parameters, hyperparams)
    │       └── traypose_env.py     # Core `TrayPoseEnv` class
    │
    ├── jointpos/                   # (Placeholder for joint position controllers)
    │
    ├── models/                     # Saved models for RL training
    │
    ├── scripts/                    # Runnable scripts
    │   ├── depthcamera/
    │   ├── liquid/
    │   ├── multi_object/
    │   └── traypose/
    │       └── visualize_traypose.py   # Demo script with MuJoCo viewer
    │
    └── training/                   # (Placeholder for RL training scripts)

🦾 Environment: TrayPoseEnv

Defined in envs/traypose/traypose_env.py.
This is a Gym-style MuJoCo environment where the Panda robot holds a tray and maintain the cylinder while moving.
🔹 State space (obs)

Observation vector includes:

    Relative cylinder position to tray center: (x, y, z)
    Tray pose: tray position (x, y, z)
    Tray yaw: orientation angle around vertical axis

→ Size = 7D vector

obs = [rel_cylinder_x, rel_cylinder_y, rel_cylinder_z,
       tray_x, tray_y, tray_z,
       tray_yaw]

🔹 Action space

Agent controls incremental changes to tray pose:

    Δx (horizontal X offset)
    Δy (horizontal Y offset)
    Δz (vertical offset)
    Δyaw (rotation around Z)

Action space: Box(-0.05, 0.05, shape=(4,))

action = [dx, dy, dz, dyaw]

🔹 Reward function

At each step:

    tray distance cost = distance between tray pose & goal pose
    yaw cost = absolute yaw error w.r.t goal yaw
    cylinder_offset cost = distance of cylinder projection from tray center

reward = -(tray_dist + yaw_error + cylinder_offset)

Episodes terminate if:

    Max steps reached (500), OR
    cylinder falls below tray (z < 0.1), OR
    Tray pose close enough to goal (goal reached)

⚙️ Setup

    git clone git@github.com:Thinkminator/franka_tray_rl.git
    cd franka_tray_rl

Setup Conda Environment:

    conda env create -f environment.yml
    conda activate franka_tray_rl

🚀 Running a Demo

Visualize the tray-cylinder environment:


    python3 scripts/traypose/visualize_traypose.py

A MuJoCo viewer will open showing the Panda arm holding the tray.
A red cylinder will spawn above the tray.
Random actions will move the tray.

🧠 Next Steps (RL Training)

You can plug TrayPoseEnv into RL libraries like Stable-Baselines3 or RLlib.


📌 TODO roadmap

- Add multi-cylinder environment (envs/multi_object)
- Add torque-sensor observation env
- Integrate RGB-D camera inputs (depthcamera env)
- Release pre-trained models