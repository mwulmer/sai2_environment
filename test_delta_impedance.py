import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from ipdb import set_trace


def main():

    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE

    env = RobotEnv(name='move_object_to_target',
                   simulation=True,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=True,
                   blocking_action=True,
                   rotation_axis=(0, 0, 1))

    obs, reward, done, info = env.reset()

    # a = np.array([0.2, -0.42, 0.15])
    # quat = env.rotvec_to_quaternion(np.array([np.pi*0.25, 0,0]))
    # a = np.concatenate([a, quat])
    #0.28673 0.0     0.55571

    a0 = np.array([0.35 - 0.28673, 0.0, 0.45 - 0.55571, 0.5, 500, 10])
    #go down to 5cm above the table
    a1 = np.array([0.0, 0.0, -0.4, 0.0, 500, 10])
    #at (0.35, 0.0, 0.05) go to one side of the possible range
    a2 = np.array([0.0, -0.2, 0.0, 0.0, 500, 10])
    #at (0.35 -0.2 0.05)
    a3 = np.array([0.0, -0.2, 0.0, 0.0, 500, 10])
    #go to 0.55,-0.4,0.05
    a4 = np.array([0.25, 0.0, 0.0, 0.0, 500, 10])

    #go to 0.6 0.4 0.05
    a5 = np.array([0.0, 0.2, 0.0, 0.1, 500, 10])
    a6 = np.array([0.0, 0.2, 0.0, 0.1, 500, 10])
    a7 = np.array([0.0, 0.2, 0.0, 0.1, 500, 10])
    a8 = np.array([0.0, 0.2, 0.0, 0.1, 500, 10])

    arr = [a0, a1, a2, a3, a4, a5, a6, a7, a8]

    for action in arr:
        obs, reward, done, info = env.step(action)

    #env.reset()


if __name__ == "__main__":
    main()
