import time
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
from gym import spaces
from ipdb import set_trace
from scipy.spatial.transform import Rotation as Rot
from sklearn.preprocessing import MinMaxScaler

from sai2_environment.client import RedisClient
from sai2_environment.action_space import *
from sai2_environment.utils import name_to_task_class, Timer
from sai2_environment.ranges import Range
from sai2_environment.camera_handler import CameraHandler


class RobotEnv(object):
    """
    The central wrapper around the robot control.
    """
    def __init__(self,
                 name='move_object_to_target',
                 simulation=True,
                 render=False,
                 action_space=ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP,
                 isotropic_gains=True,
                 blocking_action=False,
                 action_frequency=20,
                 camera_available=True,
                 rotation_axis=(True, True, True)):

        self.camera_available = camera_available
        # connect to redis server
        hostname = "127.0.0.1" if simulation else "TUEIRSI-NC-008"
        self.env_config = {
            'simulation': simulation,
            'render': render,
            'camera_resolution': (128, 128),
            'hostname': hostname,
            'port': 6379,
            'blocking_action': blocking_action,
            'rotation_axis': rotation_axis
        }

        # connect redis client
        self._client = RedisClient(config=self.env_config)
        self._client.connect()

        self.timer = Timer(frequency=action_frequency)

        self.start_time = time.time()

        # set action space to redis
        self._robot_action = get_robot_action(action_space, isotropic_gains,
                                              rotation_axis)
        #self._robot_action = RobotAction(action_space, isotropic_gains, rotation_axis=rotation_axis)

        self._client.init_action_space(self._robot_action)
        self._episodes = 0

        self.observation_space = {
            "state": self._client.get_robot_state().shape,
            "center": (3, 128, 128)
        }

        self.action_space = self._robot_action.action_space
        self.contact_event = False
        self.camera_handler = CameraHandler.getInstance(self.env_config['camera_resolution'])



        self.scaler = MinMaxScaler()
        self.scaler.fit([np.concatenate((Range.q["min"], Range.q_dot["min"], Range.tau["min"], np.zeros(1))), 
                         np.concatenate((Range.q["max"], Range.q_dot["max"], Range.tau["max"], np.ones(1)))])
        
        self.contact_thread = threading.Thread(name="contact_thread", target= self.get_contact)

        if not self.env_config["simulation"]:
            self.contact_thread.start()
            if self.camera_available:
                self.camera_handler.camera_thread.start()
        #ẃarm up camera
        time.sleep(1)

        #TODO define what all the responsibilites of task are
        task_class = name_to_task_class(name)
        self.task = task_class('tmp', self._client, camera_handler=self.camera_handler,simulation=simulation)

    def reset(self):
        self._client.reset(self._episodes)  
        #TODO do we want to set it every time or keep one action space per experiment?
        if self._episodes != 0:
            self._client.set_action_space()
       
        if self._episodes%10==0:
            print("Episode: {}; Elapsed Time: {} minutes".format(self._episodes, round((time.time()-self.start_time)/60), 4))

        self._episodes += 1
        self.task.initialize_task()        
        return self._get_obs()

    def convert_image(self, im):
        return np.rollaxis(im, axis=2, start=0)/255.0

    def get_normalized_robot_state(self):
        robot_state = self.scaler.transform([self._client.get_robot_state()])[0]
        if not self.env_config['simulation']:
            robot_state[-1] = self.contact_event
        return robot_state

    def get_contact(self):
        while True:
            contact = self._client.get_contact_occurence()
            self.contact_event = True if contact.any() else self.contact_event
            #print("contact=", contact)

    def rotvec_to_quaternion(self, vec):
        quat = Rot.from_euler('zyx', vec).as_quat()
        #[w, x, y, z]
        idx = [3, 0, 1, 2]
        return quat[idx]

    def quaternion_to_rot(self, quaternion):
        return Rot.from_quat(quaternion).as_dcm()

    def step(self, action):
        assert action.shape == self._robot_action.action_space_size(
        ), "Action shape not correct, expected shape {}".format(
            self._robot_action.action_space_size())
        #build the full action if
        action = self._robot_action.build_full_command(action)

        # blocking action waits until the action is carried out and computes reward along the trajectory
        if self.env_config['blocking_action']:
            # first check if there is still something going on on the robot
            # print("Waiting for robot: {}".format(
            #self._client.action_complete()))
            self.take_action(action)
            time.sleep(0.01)

            while not self._client.action_complete():
                time.sleep(0.01)         

            reward, done = self._compute_reward()

        # non-blocking does not wait and computes reward right away
        else:
            self.timer.wait_for_next_loop()

            self.take_action(action)
            reward, done = self._compute_reward()

        #print("Reward: {}".format(reward))
        info = None
        obs = self._get_obs() # has to be before the contact reset \!/
        #print("end of step, contact happened = ", self.contact_event)
        self.contact_event = False
        return obs, reward, done, info

    def take_action(self, action):
        return self._client.take_action(action)

    def render(self):
        return None

    def close(self):
        return 0

    def _compute_reward(self):
        reward, done = self.task.compute_reward()
        return reward, done

    def _get_obs(self):
        if self.env_config['simulation']:
            camera_frame = self.convert_image(self._client.get_camera_frame())
            robot_state = self.get_normalized_robot_state()
        else:
            camera_frame = self.convert_image(self.camera_handler.get_color_frame()) if self.camera_available else 0
            robot_state = self.get_normalized_robot_state()
        return camera_frame, robot_state
