import numpy as np
from sai2_environment.client import RedisClient
import subprocess

class RobotEnv(object):
    def __init__(self, name='pick_and_place', action_space='joint_velocities', simulation=True, render=False):
        #connect to redis server
        env_config = {'simulation': simulation, 'render': render, 'camera_resolution' : (128,128), 'hostname':"127.0.0.1", 'port': 6379}

        #connect redis client
        self._client = RedisClient(config = env_config)
        self._client.connect()

        #set action space to redis
        self._client.set_action_space(action_space)	

    def reset(self):
        return 0

    def step(self, action):        
        self._client.take_action(action)

        a = self._client.get_obs()
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def take_action(self, action):
        return self._client.take_action(action)        

    def close(self):
        return 0

    def _get_obs(self):
        obs = None
        return obs

    def _reset_sim(self):
        return 0
