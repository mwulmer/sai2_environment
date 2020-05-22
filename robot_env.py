import numpy as np
from sai2_environment.client import RedisClient
from sai2_environment.action_space import ActionSpace, RobotAction
import time
from sai2_environment.utils import name_to_task_class

from scipy.spatial.transform import Rotation as Rot


class RobotEnv(object):
    '''
    The central wrapper around the robot control. 
    '''
    def __init__(self,
                 name='move_object_to_target',
                 simulation=True,
                 render=False,
                 action_space=ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP,
                 isotropic_gains=True,
                 blocking_action=False,
                 rotate_only_z=False):
        #connect to redis server
        self.env_config = {
            'simulation': simulation,
            'render': render,
            'camera_resolution': (128, 128),
            'hostname': "127.0.0.1",
            'port': 6379,
            'blocking_action': blocking_action,
            'rotate_only_z': rotate_only_z
        }

        #connect redis client
        self._client = RedisClient(config=self.env_config)
        self._client.connect()

        #TODO define what all the responsibilites of task are
        task_class = name_to_task_class(name)
        self.task = task_class('tmp', self._client)

        #set action space to redis
        self._robot_action = RobotAction(action_space, isotropic_gains)

        self._client.set_action_space(self._robot_action)
        self._reset_counter = 0

    def reset(self):
        #need to reset simulator different from robot
        #these functions wait for the environemnt(simulation) or real robot to be reset
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
        quat = Rot.from_euler('zyx', vec).as_quat()
        #[w, x, y, z]
        idx = [3, 0, 1, 2] 
        return quat[idx] 

    def quaternion_to_rot(self, quaternion):
        return Rot.from_quat(quaternion).as_dcm()

    def step(self, action):
        if(self.env_config['rotate_only_z']):
            x = action[:3]
            rot = np.pi*action[3]            
            quat = self.rotvec_to_quaternion(np.array([rot, 0, 0]))
            print(quat)
            stiffness = action[4:]
            action = np.concatenate([x, quat, stiffness])

        assert action.shape == self._robot_action.action_space_size(
        ).shape, "Action shape not correct, expected shape {}".format(self._robot_action.action_space_size(
        ).shape)
        #blocking action waits until the action is carried out and computes reward along the trajectory
        if self.env_config['blocking_action']:
            #first check if there is still something going on on the robot
            print("Waiting for robot: {}".format(
                self._client.action_complete()))
            while not self._client.action_complete():
                time.sleep(0.01)

            self.take_action(action)
            time.sleep(0.01)

            while not self._client.action_complete():
                time.sleep(0.01)

            reward, done = self._compute_reward()

        #non-blocking does not wait and computes reward right away
        else:
            self.take_action(action)
            reward, done = self._compute_reward()

        print("Reward: {}".format(reward))
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
        camera_frame = self._client.get_camera_frame()
        robot_state = self._client.get_robot_state()
        return camera_frame, robot_state
