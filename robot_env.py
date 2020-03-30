import numpy as np
from client import RedisClient
import subprocess

class RobotEnv(object):
    def __init__(self, name='pick_and_place', action_space='joint_velocities', simulation=True,render=False):
        #connect to redis server
        env_config = {'simulation': simulation, 'render': render, 'camera_resolution' : (128,128), 'hostname':"127.0.0.1", 'port': 6379}

        if simulation:
            subprocess.call(['./simviz01'], cwd='/home/maxwulmer/sai2/apps/FrankaApp/bin/01-pick_and_place/')

        self._client = RedisClient(config = env_config)
        self._client.connect()	

    def reset(self):
        return 0

    def step(self, action):
        a = self._client.get_obs()

        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def close(self):
        return 0

    def _get_obs(self):
        obs = None
        return obs

    def _reset_sim(self):
        return 0
