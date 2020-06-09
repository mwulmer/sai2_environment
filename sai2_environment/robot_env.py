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
from sai2_environment.utils import name_to_task_class
from sai2_environment.ranges import Range


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
                 blocking_time=2.0,
                 rotation_axis=(True, True, True)):

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

        #TODO define what all the responsibilites of task are
        task_class = name_to_task_class(name)
        self.blocking_time = blocking_time
        self.task = task_class('tmp', self._client, simulation=simulation)

        # set action space to redis
        self._robot_action = get_robot_action(action_space, isotropic_gains,
                                              rotation_axis)
        #self._robot_action = RobotAction(action_space, isotropic_gains, rotation_axis=rotation_axis)

        self._client.set_action_space(self._robot_action)
        self._reset_counter = 0

        self.observation_space = {
            "state": self._client.get_robot_state().shape,
            "center": (3, 128, 128)
        }
        self.action_space = self._robot_action.action_space
        self.pipeline = rs.pipeline()
        self.color_frame = None
        self.depth_frame = None

        self.scaler = MinMaxScaler()
        self.scaler.data_min_ = np.concatenate((Range.q["min"], Range.q_dot["min"], Range.tau["min"]))
        self.scaler.data_max_ = np.concatenate((Range.q["max"], Range.q_dot["max"], Range.tau["max"]))

        self.background = threading.Thread(name="background", target= self.get_frames)
        if not self.env_config["simulation"]:
            self.background.start()

    def reset(self):
        # need to reset simulator different from robot
        # these functions wait for the environemnt(simulation) or real robot to be reset
        if not self.env_config['simulation'] or (self._reset_counter == 0):
            '''
            bring the robot back to its initial state if we are in the real world 
            OR if its the first reset (meaning that sim and controller have just been started)
            '''
            self._client.reset_robot()
        else:
            # if in simulation, hard reset both simulator and controller
            self._client.env_hard_reset()        
        print("--------------------------------------")
        self._reset_counter += 1
        return self._get_obs()

    def convert_image(self, im):
        return np.rollaxis(im, axis=2, start=0)/255.0

    def get_normalized_robot_state(self):
        return self.scaler.transform(self._client.get_robot_state())

    def get_frames(self):
        self.pipeline.start()
        align_to = rs.stream.color
        align = rs.align(align_to)
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            self.depth_frame = np.asanyarray(depth_frame.get_data())
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            self.color_frame = cv2.resize(color_frame, self.env_config['camera_resolution'])

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
            t0 = time.time()

            waited_time = 0
            while not self._client.action_complete():
                time.sleep(0.01)
                if time.time()-t0 > self.blocking_time:
                    break

            reward, done = self._compute_reward()

        # non-blocking does not wait and computes reward right away
        else:
            self.take_action(action)
            reward, done = self._compute_reward()

        #print("Reward: {}".format(reward))
        info = None
        return self._get_obs(), reward, done, info

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
            camera_frame = self.convert_image(self.color_frame)
            robot_state = self.get_normalized_robot_state()
        return camera_frame, robot_state
