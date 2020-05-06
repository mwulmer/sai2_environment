import redis
import numpy as np
import json
import time
import sys

class RedisClient(object):
    def __init__(self, config):
        self.config = config
        self._hostname = self.config['hostname']
        self._port = self.config['port']
        self._camera_resolution = self.config['camera_resolution']
        self._sim = self.config['simulation']
        self._conn = None

        #TODO there certainly is a nicer way to do this
        #init keys to read/write to redis server
        self.ACTION_SPACE_KEY = "sai2::ReinforcementLearning::action_space"
        self.ACTION_KEY = "sai2::ReinforcementLearning::action"
        self.START_ACTION_KEY = "sai2::ReinforcementLearning::start_action"
        self.CAMERA_DATA_KEY  = "sai2::ReinforcementLearning::camera_data"
        self.ROBOT_IS_RESET_KEY = "sai2::ReinforcementLearning::robot_is_reset"      
        self.ACTION_COMPLETE_KEY =   "sai2::ReinforcementLearning::action_complete" 

        if self._sim:
            self.JOINT_TORQUES_COMMANDED_KEY = "sai2::PandaApplication::actuators::fgc"
            self.JOINT_ANGLES_KEY  = "sai2::PandaApplication::sensors::q"
            self.JOINT_VELOCITIES_KEY = "sai2::PandaApplication::sensors::dq"

            self.SENSED_CONTACT_KEY = "sai2::PandaApplication::sensors::contact"

            self.HARD_RESET_CONTROLLER_KEY = "sai2::ReinforcementLearning::hard_reset_controller"
            self.HARD_RESET_SIMULATOR_KEY = "sai2::ReinforcementLearning::hard_reset_simulator"

        else:        
            self.JOINT_TORQUES_COMMANDED_KEY = "sai2::FrankaPanda::actuators::fgc"
            self.JOINT_ANGLES_KEY  = "sai2::FrankaPanda::sensors::q"
            self.JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::sensors::dq"
            self.MASSMATRIX_KEY = "sai2::FrankaPanda::sensors::model::massmatrix"
            self.CORIOLIS_KEY = "sai2::FrankaPanda::sensors::model::coriolis"
            self.ROBOT_GRAVITY_KEY = "sai2::FrankaPanda::sensors::model::robot_gravity"

    def connect(self):
        try:
            self._conn = redis.StrictRedis(
                host=self._hostname,
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
        data = self.redis2array(self.get(self.CAMERA_DATA_KEY)) 
        (w, h) = self._camera_resolution  
        b = np.reshape(data[0::3], (w,h))
        g = np.reshape(data[1::3], (w,h))
        r = np.reshape(data[2::3], (w,h))
        frame = np.flip((np.dstack((r,g,b))).astype(np.uint8), 0)
        return frame
    
    def get_robot_state(self) -> np.array:
        q = self.redis2array(self.get(self.JOINT_ANGLES_KEY))
        dq = self.redis2array(self.get(self.JOINT_VELOCITIES_KEY))
        tau = self.redis2array(self.get(self.JOINT_TORQUES_COMMANDED_KEY))
        #TODO add contact
        #contact = self.redis2array(self.get(self.SENSED_CONTACT_KEY))
        #print(contact)
        #TODO EE Pose

        return np.concatenate([q, dq, tau])

    def redis2array(self, serialized_arr: str) -> np.array:
        return np.array(json.loads(serialized_arr))

    def take_action(self, action):
        self.set(self.ACTION_KEY, self.array2redis(action))    
        return self.set(self.START_ACTION_KEY, 1)  

    def set_action_space(self, action_space):
        return self.set(self.ACTION_SPACE_KEY, action_space.value)

    def array2redis(self, arr: np.array) -> str:
        return json.dumps(arr.tolist())
    
    def robot_is_reset(self) -> bool:
        return int(self.get(self.ROBOT_IS_RESET_KEY).decode()) == 1
    
    def action_complete(self) -> bool:
        return int(self.get(self.ACTION_COMPLETE_KEY).decode()) == 1

    def reset_robot(self) -> bool:
        self.take_action(np.array([-1, -1, -1, -1, -1, -1, -1]))
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
        self.set(self.HARD_RESET_CONTROLLER_KEY, 1)
        self.set(self.HARD_RESET_SIMULATOR_KEY, 1)

        controller_reset = False
        simulator_reset = False
        waited_time = 0
        print("[INFO] Waiting for the simulator and controller to reset")
        while controller_reset and simulator_reset:
            controller_reset = int(self.get(self.HARD_RESET_CONTROLLER_KEY).decode()) == 0
            simulator_reset = int(self.get(self.HARD_RESET_SIMULATOR_KEY).decode()) == 0

            time.sleep(0.1)
            #if we have to wait for more than a minute something went wrong
            waited_time += 0.1
            if waited_time > 60:
                sys.exit(0)
                return False
        #TODO move this to logging
        print("[INFO] Successfully reset simulator and controller!")
        return True


    def get(self, key):
        return self._conn.get(key)

    def set(self, key, value):
        return self._conn.set(key, value)

    def delete(self, key):
        self._conn.delete(key)


