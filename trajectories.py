import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
import cv2


def main():

    action_space = ActionSpace.ABS_EE_POSE_IMPEDANCE

    env = RobotEnv(name='move_object_to_target',
                   simulation=False,
                   action_space=action_space,
                   blocking_action=True,
                   # action_frequency=20,
                   camera_available=True,
                   rotation_axis=(1, 1, 1))

    episodes = 20
    steps = 1000

    start_time = time.time()

    obs = env.reset()

    # a = np.array([0.2, -0.42, 0.15])
    # quat = env.rotvec_to_quaternion(np.array([np.pi*0.25, 0,0]))
    # a = np.concatenate([a, quat])
    # 0.28673 0.0     0.55571
    arr = []

    a0 = np.array([0.5, 0.0, 0.3, 0, 1, 0, 0, 500, 15])
    arr.append(a0)
    # a1 = np.array([0.4, 0.0, 0.2, 0, 1, 0, 0, 500, 15])
    # arr.append(a1)
    # a2 = np.array([0.4, 0.0, 0.1, 0, 1, 0, 0, 500, 15])    
    # arr.append(a2)   
    # a3 = np.array([0.4, 0.0, 0.05, 0, 1, 0, 0, 500, 15])
    # arr.append(a3) 
    # a4 = np.array([0.4, 0.0, 0.02, 0, 1, 0, 0, 500, 15])
    # arr.append(a4) 
    # a5 = np.array([0.4, 0.0,  0.015, 0, 1, 0, 0, 500, 15])
    # arr.append(a5)
    # a6 = np.array([0.4, 0.2,  0.015, 0, 1, 0, 0, 500, 15])
    # arr.append(a6) 
    # a7 = np.array([0.4, 0.3,  0.015, 0, 1, 0, 0, 500, 15])
    # arr.append(a7) 

    for action in arr:
        obs, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
