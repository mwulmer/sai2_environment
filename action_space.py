"""
Adapted from https://github.com/stepjam/RLBench
"""
from enum import Enum

class ActionSpace(Enum):
    # Absolute arm joint positions/angles (in radians)
    ABS_JOINT_POSITION = 1

    # Change in arm joint positions/angles (in radians)
    DELTA_JOINT_POSITION = 2

    # Absolute arm joint forces/torques
    ABS_JOINT_TORQUE = 3

    # Change in arm joint forces/torques
    DELTA_JOINT_TORQUE = 4

    # Absolute end-effector pose (position (3) and quaternion (4))
    ABS_EE_POSE = 5

    # Change in end-effector pose (position (3) and quaternion (4))
    DELTA_EE_POSE = 6

    #TODO Impedance Control for EE_Pose and other space



