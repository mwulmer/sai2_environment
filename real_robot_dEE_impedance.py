import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

def main():

    action_space = ActionSpace.MT_EE_POSE_IMPEDANCE

    env = RobotEnv(name='move_object_to_target',
                   simulation=False,
                   action_space=action_space,
                   action_frequency=20,
                   camera_available=False,
                   rotation_axis=(0, 0, 0))    

    episodes = 20
    steps = 2000

    start_time = time.time()    
    writer = SummaryWriter()

    for episode in range(episodes):
        
        print("Episode: {}; Elapsed Time: {} minutes".format(episode, round((time.time()-start_time)/60), 4))
        obs = env.reset()
        for step in range(steps):
            position = np.random.normal(loc=0.0, scale=0.1, size=(3,))
            stiffness_linear =  np.random.normal(loc=0.0, scale=10, size=(1,))
            stiffness_rot = np.random.normal(loc=0.0, scale=1, size=(1,))
            action = np.concatenate((position,stiffness_linear,stiffness_rot))

            obs, reward, done, info = env.step(action)
            print (reward)
            
    
    print("Action frequency: {}".format(env.timer._update_counter/(time.time()-start_time)))

if __name__ == "__main__":
    main()
