# Franka Tray RL

![alt text](image.png)

This repository contains a custom MuJoCo + Gym environment for robotic manipulation with a Franka Panda arm, where the robot holds a tray and must balance/manipulate a cylinder for reinforcement learning research.

    📂 Repository Structure
    franka_tray_rl/
    │
    ├── assets/                     # Models & meshes
    │   ├── panda_tray/             # Panda + tray assets
    │   │   ├── mesh
    │   │   ├── testure
    │   │   ├── panda_tray.urdf     # URDF definition of Panda + Tray
    │   │   ├── panda.xml # MJCF (MuJoCo) of robot
    |       ├── world.xml # MJCF (MuJoCo) of floor, robot, cylinder and markers
    |       ├── position_PID.py # python script of moving each joint
    |       ├── demo.py # python script of fix arm at a pose
    |       ├── action_constant.py  # python script for zero, seeded, random action (same value)
    |       └── action_changing.py  # python script for zero, seeded, random action (changing value)
    │
    ├── debug/                      # Standalone debugging scripts
    │   ├── check_urdf_xml.py       # Check whether urdf and xml is the same - essential for using URDF with pybullet to get IK solutions for XML
    │   ├── fk_generator.py         # Generates forward kinematics
    |   ├── ik_generator.py         # Generates inverse kinematics
    │   ├── interactive_ik.py       # Interactive IK exploration
    │   ├── urdf_viewer.py          # Loads URDF in viewer
    │   └── xml_viewer.py           # Loads XML in MuJoCo viewer
    │
    ├── docs/  
    │   └── TrayPose.md             # Explain TrayPose environment
    |
    ├── matlab/
    ├── Network 
    ├── envs/                       # OpenAI Gym environments
    │   └── traypose/               # Main tray manipulation environment
    │       ├── __init__.py
    │       ├── config.yaml         # Config file (parameters)
    │       └── traypose_env.py     # Core `TrayPoseEnv` class
    │
    ├── jointpos/                   # Placeholder for joint position values to define start & goal pose
    │   └── config.txt              # Tested start and goal position info
    ├── scripts/                    # Runnable scripts
    │   └── traypose/
    |       ├── test_trainpose.py       # Test train models
    |       ├── train_traypose_tensorboard.py       # Train models
    │       └── visualize_traypose.py   # Demo script with MuJoCo viewer
    │
    └── training/                   # Folder for RL training data

## 🦾 Environment: 
### 1. TrayPoseEnv
- [Document](docs/TrayPose.md)
- [Code](envs/traypose/traypose_env.py)

## ⚙️ Setup

    git clone git@github.com:Thinkminator/franka_tray_rl.git
    cd franka_tray_rl

Setup Conda Environment:

    conda env create -f environment.yml
    conda activate franka_tray_rl

## 🚀 Running a Env Demo

Visualize the tray-cylinder environment:


    python scripts/traypose/visualize_traypose.py [Mode]


- [Mode] = zero, seeded or random 
    - zero: Zero action mode (arm stays at start pose)
    - random_changing: Random and changing action mode (arm moves randomly)
    - random_constant: Random but same action mode (arm moves randomly)
    - seeded_constant: Same random actions with fixed RNG seed for reproducibility
    - seeded_changing: Changing random actions with fixed RNG seed for reproducibility

A MuJoCo viewer will open showing the Panda arm holding the tray.
A red cylinder will spawn above the tray.
Random actions will move the tray.

## 🧠 Solve with Reinforcement Learning
### Running a Network model (No viewer render)

To see a forward and backward pass with network update:

    python Network/Net_demo.py
    
### Training a model(Pytorch based, no viewer render)

Perform training loop with Pytorch:

    python Network/Torch_train.py

### Training a model (SB) with Stable Baseline 3 

Evaluate with TensorBoard:
Save the model every 100000 steps at phase 5:

    python scripts/traypose/train_traypose_tensorboard.py

Save the model from phase 1, each phase saves 1 model:

    python scripts/traypose/train_traypose_tensorboard_each.py

To see the training evaluation:

    tensorboard --logdir training/logs


### Testing a model (SB3, with viewer render)

To evaluate a model:

    python scripts/traypose/test_trainpose.py

To change the phase, just modify the env.set_phase([phase]), which [phase] will be 1-5.

## 🛠️ Solve with STOMP
1. Launch MATLAB and open the matlab folder
2. Open the [panda_tray_STOMP_Path_Planning.mlx](matlab/panda_tray_STOMP_Path_Planning.mlx) and press "Run"


## 🔍 Result Viewing
1. The tensorboard graphs are stored in [tensorboard_graph](tensorboard_graph).
2. The recordings of RL models and STOMP performances are kept at [videos](videos).
