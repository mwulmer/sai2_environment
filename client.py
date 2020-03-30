import redis
import numpy as np
import json

class RedisClient(object):
    def __init__(self, config):
        self.config = config
        self._hostname = self.config['hostname']
        self._port = self.config['port']
        self._camera_resolution = self.config['camera_resolution']
        self._sim = self.config['simulation']
        self._conn = None

        #init keys to read/write to redis server
        self.ACTION_SPACE_KEY = "sai2::ReinforcementLearning::ActionSpace"
        self.ACTION_KEY = "sai2::ReinforcementLearning::Action"
        self.CAMERA_DATA_KEY  = "sai2::ReinforcementLearning::camera_data"

        if self._sim:
            self.JOINT_TORQUES_COMMANDED_KEY = "sai2::DemoApplication::Panda::actuators::fgc"
            self.JOINT_ANGLES_KEY  = "sai2::DemoApplication::Panda::sensors::q"
            self.JOINT_VELOCITIES_KEY = "sai2::DemoApplication::Panda::sensors::dq"

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

    def get_obs(self):
        q = self.get(self.JOINT_ANGLES_KEY)
        dq =self.get(self.JOINT_VELOCITIES_KEY)

    def get_camera_frame(self) -> np.array:
        data = self.redis2array(self.get('sai2::ReinforcementLearning::camera_data')) 
        (w, h) = self._camera_resolution  
        r = np.reshape(data[0::3], (w,h))
        g = np.reshape(data[1::3], (w,h))
        b = np.reshape(data[2::3], (w,h))
        frame = np.flip((np.dstack((r,g,b))).astype(np.uint8), 0)
        return frame        

    def redis2array(self, serialized_arr: str) -> np.array:
        return np.array(json.loads(serialized_arr))

    def array2redis(self, arr: np.array) -> str:
        return json.dumps(arr.tolist())

    def get(self, key):
        return self._conn.get(key)

    def set(self, key, value):
        return self._conn.set(key, value)

    def delete(self, key):
        self._conn.delete(key)


