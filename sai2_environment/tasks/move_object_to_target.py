from sai2_environment.tasks.task import Task
import numpy as np

class MoveObjectToTarget(Task):
    def __init__(self, task_name, redis_client,camera_handler,simulation=True):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.camera_handler = camera_handler
        self.TARGET_OBJ_POSITION_KEY  = "sai2::ReinforcementLearning::move_object_to_target::object_position"
        self.GOAL_POSITION_KEY  = "sai2::ReinforcementLearning::move_object_to_target::goal_position"

        if simulation:
            self.goal_position = self._client.redis2array(self._client.get(self.GOAL_POSITION_KEY))
            self.current_obj_position = self.get_current_position()
            self.last_obj_position = self.current_obj_position
            self.total_distance = self.euclidean_distance(self.goal_position, self.current_obj_position)
        else:
            #setup the things that we need in the real world
            # self.goal_position = None
            # self.current_obj_position = None
            # self.last_obj_position = None
               
            # new modify
            self.current_obj_distance = self.camera_handler.grab_distance()
            self.last_obj_distance = self.current_obj_distance
            self.total_distance = self.camera_handler.grab_distance()


    def compute_reward(self):
        if self._simulation:
            self.last_obj_position = self.current_obj_position
            self.current_obj_position = self.get_current_position()
            d0 = self.euclidean_distance(self.goal_position, self.last_obj_position)
            d1 = self.euclidean_distance(self.goal_position, self.current_obj_position)

            reward = (d0 - d1)/self.total_distance
            #radius of target location is 0.04
            done = np.linalg.norm(self.goal_position - self.current_obj_position) < 0.04
        else:
            # reward = 0
            #TODO
            # new modify
            self.last_obj_distance = self.current_obj_distance
            self.current_obj_distance = self.camera_handler.grab_distance()
            d_last = self.last_obj_distance
            d_current = self.current_obj_distance
             # When detecting no enough markers at the very beginning
            if d_current==1:
                reward = 0
            else:
                reward = (d_last - d_current)/self.total_distance
                # reward = d_current
            done = d_current<0.04
        return self.camera_handler.grab_distance(), done

    def initialize_task(self):
        if self._simulation:
            self.goal_position = self._client.redis2array(self._client.get(self.GOAL_POSITION_KEY))
            self.current_obj_position = self.get_current_position()
            self.last_obj_position = self.current_obj_position
            self.total_distance = self.euclidean_distance(self.goal_position, self.current_obj_position)
        else:
            self.total_distance = self.camera_handler.grab_distance()



    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_current_position(self):
        return self._client.redis2array(self._client.get(self.TARGET_OBJ_POSITION_KEY))

