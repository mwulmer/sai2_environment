import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time


def main():

    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE

    env = RobotEnv(name='move_object_to_target',
                   simulation=False,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=True,
                   blocking_action=True,
                   rotation_axis=(0, 0, 1))    

    episodes = 50
    steps = 100

    for episode in range(episodes):
        obs, reward, done, info = env.reset()
        for step in range(steps):
            position = np.random.uniform(low=-0.1, high=0.1, size=(3,))
            rotation = np.random.uniform(low=-0.2, high=0.2, size=(1,))
            stiffness_linear = np.random.uniform(low=-50, high=50, size=(1,))
            stiffness_rot = np.random.uniform(low=-2, high=2, size=(1,))
            action = np.concatenate((position,rotation,stiffness_linear,stiffness_rot))

            obs, reward, done, info = env.step(action)



if __name__ == "__main__":
    main()
