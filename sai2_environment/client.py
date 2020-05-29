import redis
import numpy as np
import json
import time
import sys
from sai2_environment.redis_keys import RedisKeys


class RedisClient(object):
    def __init__(self, config):
        self._config = config
        self._hostname = self._config['hostname']
        self._port = self._config['port']
        self._camera_resolution = self._config['camera_resolution']
        self._sim = self._config['simulation']

        self._conn = None
        self.keys = RedisKeys(self._sim)

        self._action_space = None
        self._action_space_size = None
        self._reset_action = None

    def connect(self):
        try:
            self._conn = redis.StrictRedis(host=self._hostname,
                                           port=self._port)
            print(self._conn)
            self._conn.ping()
            print('Connected to Redis Server')
        except Exception as ex:
            print('Error: {}'.format(ex))
            exit('Failed to connect, terminating')

    def ping(self):
        self._conn.ping()

    def get_camera_frame(self) -> np.array:
        data = self.redis2array(self.get(self.keys.CAMERA_DATA_KEY))
        (w, h) = self._camera_resolution
        b = np.reshape(data[0::3], (w, h))
        g = np.reshape(data[1::3], (w, h))
        r = np.reshape(data[2::3], (w, h))
        frame = np.flip((np.dstack((r, g, b))).astype(np.uint8), 0)
        return frame

    def get_robot_state(self) -> np.array:
        q = self.redis2array(self.get(self.keys.JOINT_ANGLES_KEY))
        dq = self.redis2array(self.get(self.keys.JOINT_VELOCITIES_KEY))
        tau = self.redis2array(self.get(
            self.keys.JOINT_TORQUES_COMMANDED_KEY))

        if self._config["simulation"]:
            contact = self.redis2array(self.get(self.keys.SENSED_CONTACT_KEY))
        else:
            #TODO No force sensor on robot, need to use the sensed torques 
            contact = np.array([0])

        return np.append(np.concatenate([q, dq, tau]), contact)

    def redis2array(self, serialized_arr: str) -> np.array:
        return np.array(json.loads(serialized_arr))

    def take_action(self, action):
        self.set(self.keys.ACTION_KEY, self.array2redis(action))
        return self.set(self.keys.START_ACTION_KEY, 1)

    def set_action_space(self, robot_action):
        self._action_space = robot_action.action_space_enum
        #TODO this will send the wrong action size right now
        self._action_space_size = robot_action.action_space_size()
        self._reset_action = robot_action.reset_action()
        return self.set(self.keys.ACTION_SPACE_KEY, self._action_space.value)

    def array2redis(self, arr: np.array) -> str:
        return json.dumps(arr.tolist())

    def robot_is_reset(self) -> bool:
        return int(self.get(self.keys.ROBOT_IS_RESET_KEY).decode()) == 1

    def action_complete(self) -> bool:
        return int(self.get(self.keys.ACTION_COMPLETE_KEY).decode()) == 1

    def reset_robot(self) -> bool:
        self.take_action(self._reset_action)
        robot_is_reset = self.robot_is_reset()
        waited_time = 0
        if not robot_is_reset:
            print("[INFO] Waiting for the robot to reset")
            while not robot_is_reset:
                robot_is_reset = self.robot_is_reset()
                time.sleep(0.1)

                waited_time += 0.1
                #if we have to wait for more than a minute something went wrong
                if waited_time > 60:
                    sys.exit(0)
                    return False
        #TODO move this to logging
        print("[INFO] Successfully moved the robot to its initial state!")
        return True

    def env_hard_reset(self) -> bool:
        self.set(self.keys.HARD_RESET_CONTROLLER_KEY, 1)
        self.set(self.keys.HARD_RESET_SIMULATOR_KEY, 1)

        controller_reset = False
        simulator_reset = False
        waited_time = 0
        print("[INFO] Waiting for the simulator and controller to reset")
        while controller_reset and simulator_reset:
            controller_reset = int(
                self.get(self.keys.HARD_RESET_CONTROLLER_KEY).decode()) == 0
            simulator_reset = int(
                self.get(self.keys.HARD_RESET_SIMULATOR_KEY).decode()) == 0

            time.sleep(0.1)
            #if we have to wait for more than a minute something went wrong
            waited_time += 0.1
            if waited_time > 60:
                sys.exit(0)
                return False
        #TODO move this to logging
        print("[INFO] Successfully reset simulator and controller!")

        #send the reset action again such that the controller knows the current action space
        self.take_action(self._reset_action)
        self.set(self.keys.ACTION_SPACE_KEY, self._action_space.value)
        return True

    def get(self, key):
        return self._conn.get(key)

    def set(self, key, value):
        return self._conn.set(key, value)

    def delete(self, key):
        self._conn.delete(key)

