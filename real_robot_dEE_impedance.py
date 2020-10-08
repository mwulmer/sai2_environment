from sai2_environment.robot_env import RobotEnv
from sai2_environment.utils.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque


def main():

    action_space = ActionSpace.MT_EE_POSE_IMPEDANCE

<<<<<<< HEAD
    env = RobotEnv(name='move_object_to_target',
                   simulation=False,
                   action_space=action_space,
                   action_frequency=20,
                   render=True,
                   camera_available=True,
                   rotation_axis=(0, 0, 0))    

    episodes = 1
    steps = 500
=======
    env = RobotEnv(
        domain_name="reach_site",
        task_name="easy",
        simulation=True,
        action_space=action_space,
        action_frequency=20,
        render=False,
        camera_available=True,
        camera_res=(128, 128),
        rotation_axis=(0, 0, 0),
    )

    episodes = 10
    steps = 200
>>>>>>> master

    start_time = time.time()

    for episode in range(episodes):
<<<<<<< HEAD
        
        print("Episode: {}; Elapsed Time: {} minutes".format(episode, round((time.time()-start_time)/60), 4))
        
=======

        print(
            "Episode: {}; Elapsed Time: {} minutes".format(
                episode, round((time.time() - start_time) / 60), 4
            )
        )
>>>>>>> master
        obs = env.reset()
        
        
        acc_reward = 0
        for step in range(steps):
            position = np.random.normal(loc=0.0, scale=0.1, size=(3,))
            stiffness_linear = np.random.normal(loc=0.0, scale=10, size=(1,))
            stiffness_rot = np.random.normal(loc=0.0, scale=1, size=(1,))
<<<<<<< HEAD
            action = np.concatenate((position,stiffness_linear,stiffness_rot))
            # action = env.act_optimally()
            # print(action)

            # obs, reward, done, info = env.step(action)
            # obs = env.reset()
            # if done:                
            #     obs = env.reset()
            #     continue
                
=======
            action = np.concatenate(
                (position, stiffness_linear, stiffness_rot)
            )
            action = env.act_optimally()

            obs, reward, done, info = env.step(action)
            env.render(mode="human")
            acc_reward += reward

            if done:
                # print(acc_reward)
                # obs = env.reset()
                continue

>>>>>>> master
            # if render:
            #     im = obs[0]
            #     if im is not None:
            #         im = np.rollaxis(im, 0, 3)
            #         cv2.imshow('RealSense',im)
            #     key = cv2.waitKey(1)
            #     if key & 0xFF == ord('q') or key == 27:
            #         cv2.destroyAllWindows()
            #         break
<<<<<<< HEAD
        # env.close()
=======
>>>>>>> master

    print(
        "Action frequency: {}".format(
            env.timer._update_counter / (time.time() - start_time)
        )
    )


if __name__ == "__main__":
    main()
