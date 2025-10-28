import os
import sys
import argparse
import time  # <-- For sleep in viewer loop
import numpy as np
import gymnasium as gym
import mujoco.viewer  # <-- Import viewer

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
