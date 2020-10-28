from sai2_environment.robot_env import RobotEnv
from sai2_environment.utils.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import deque


def main():

    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE

    env = RobotEnv(
        domain_name="reach_site",
        task_name="easy",
        simulation=True,
        action_space=action_space,
        action_frequency=10,
        render=False,
        camera_available=True,
        rotation_axis=(0, 0, 0),
        mod_shapes=dict(
            cam=(3, 84, 84), x=(3,), dx=(3,), q=None, dq=None, tau=None
        ),
    )

    episodes = 10
    steps = 1000

    start_time = time.time()
    for episode in range(episodes):

        print(
            "Episode: {}; Elapsed Time: {} minutes".format(
                episode, round((time.time() - start_time) / 60), 4
            )
        )
        obs = env.reset()
        acc_reward = 0
        for step in range(steps):
            position = np.random.normal(loc=0.0, scale=0.2, size=(3,))
            # position = np.array([0, 0, 0])
            stiffness_linear = np.random.normal(loc=0.0, scale=10, size=(1,))
            stiffness_rot = np.random.normal(loc=0.0, scale=1, size=(1,))
            action = np.concatenate(
                (position, stiffness_linear, stiffness_rot)
            )
            # action = env.act_optimally()

            obs, reward, done, info = env.step(action)
            print(obs["cam"].shape, obs["x"].shape, obs["dx"].shape)
            # env.render(mode="human")
            acc_reward += reward

            if done:
                continue

    print(
        "Action frequency: {}".format(
            env.timer._update_counter / (time.time() - start_time)
        )
    )


if __name__ == "__main__":
    main()
