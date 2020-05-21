from sai2_environment.tasks.task import Task
import numpy as np

class MoveObjectToTarget(Task):
    def __init__(self, task_name, redis_client):
        self.task_name = task_name
        self.client = redis_client
        self.TARGET_OBJ_POSITION_KEY  = "sai2::ReinforcementLearning::move_object_to_target::object_position"
        self.GOAL_POSITION_KEY  = "sai2::ReinforcementLearning::move_object_to_target::goal_position"

        self.goal_position = self.client.redis2array(self.client.get(self.GOAL_POSITION_KEY))
        self.current_obj_position = self.get_current_position()
        self.last_obj_position = self.current_obj_position
        self.total_distance = self.euclidean_distance(self.goal_position, self.current_obj_position)
        print("Total possible distance: {}".format(self.total_distance))

    def compute_reward(self):
        self.last_obj_position = self.current_obj_position
        self.current_obj_position = self.get_current_position()
        d0 = self.euclidean_distance(self.goal_position, self.last_obj_position)
        d1 = self.euclidean_distance(self.goal_position, self.current_obj_position)

        reward = (d0 - d1)/self.total_distance
        #radius of target location is 0.04
        done = np.linalg.norm(self.goal_position - self.current_obj_position) < 0.04
        return reward, done

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_current_position(self):
        return self.client.redis2array(self.client.get(self.TARGET_OBJ_POSITION_KEY))

