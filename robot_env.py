import numpy as np
from sai2_environment.client import RedisClient
from sai2_environment.action_space import ActionSpace
import time
from sai2_environment.utils import name_to_task_class

from scipy.spatial.transform import Rotation as Rot

class RobotEnv(object):
    def __init__(self, name='move_object_to_target', action_space=ActionSpace.ABS_JOINT_POSITION, simulation=True, render=False, blocking_action=False):
        #connect to redis server
        self.env_config = {'simulation': simulation, 'render': render, 'camera_resolution' : (128,128), 'hostname':"127.0.0.1", 'port': 6379, 'blocking_action': blocking_action}

        #connect redis client
        self._client = RedisClient(config = self.env_config)
        self._client.connect()

        #TODO define what all the responsibilites of task are
        task_class = name_to_task_class(name)
        self.task = task_class('tmp', self._client)

        #set action space to redis
        self._client.set_action_space(action_space)
        self._reset_counter = 0

    def reset(self):
        #need to reset simulator different from robot
        # these functions wait for the environemnt(simulation) or real robot to be reset
        if not self.env_config['simulation'] or (self._reset_counter == 0):
            '''
            bring the robot back to its initial state if we are in the real world 
            OR if its the first reset (meaning that sim and controller have just been started)
            '''        
            self._client.reset_robot()
        else:
            #if in simulation, hard reset both simulator and controller
            self._client.env_hard_reset()
            
        print("-------------------------------------")
        print("[INFO] Robot state is reset")
        self._reset_counter += 1
        reward = None
        done = False
        info = None
        return self._get_obs(), reward, done, info

    def rotvec_to_quaternion(self, vec):
        return Rot.from_euler('zyx', vec).as_quat()

    def quaternion_to_rot(self, quaternion):
        return Rot.from_quat(quaternion).as_dcm()

    def step(self, action):
        #blocking action waits until the action is carried out and computes reward along the trajectory
        if self.env_config['blocking_action']:        
            self.take_action(action)
            time.sleep(0.01) 
            action_complete = self._client.action_complete()
            print("Blocking action")

            while not action_complete:
                time.sleep(0.01) 
                action_complete = self._client.action_complete()                                   

            reward, done = self.compute_reward()
            
        #non-blocking does not wait and computes reward right away
        else:
            self.take_action(action)
            reward, done = self.compute_reward()

        print(reward)    
        info = None
        return self._get_obs(), reward, done, info

    def take_action(self, action):
        return self._client.take_action(action)

    def compute_reward(self):
        reward, done = self.task.compute_reward()
        #TODO move this to a task specific class

        return reward, done

    def close(self):
        return 0

    def _get_obs(self):
        camera_frame = self._client.get_camera_frame()
        robot_state = self._client.get_robot_state()
        return camera_frame, robot_state

    def _reset_sim(self):
        return 0
