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
                   simulation=True,
                   action_space=action_space,
                   blocking_action=True,
                   # action_frequency=20,
                   camera_available=False,
                   rotation_axis=(1, 1, 1))

    episodes = 20
    steps = 500

    start_time = time.time()

    obs = env.reset()

    while True:
        x = np.random.uniform(low=0.3, high=0.55, size=(1,))
        y = np.random.uniform(low=-0.4, high=0.4, size=(1,))
        z = np.array([0.05])
        q = np.array([0,1,0,0])
        stiff = np.array([500, 15])
        action = np.concatenate((x,y,z,q,stiff))
    
        obs, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()