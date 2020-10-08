from sai2_environment.tasks.task import Task
import numpy as np
import psutil

np.set_printoptions(precision=3, suppress=True)


class PushPuckEasy(Task):
    def __init__(
        self, task_name, redis_client, camera_handler, simulation=True
    ):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.camera_handler = camera_handler

        # If simulated environment, check if simulation process is running
        if self._simulation:
            assert "sim01-push_puck_easy" in (
                p.name() for p in psutil.process_iter()
            ), "Simulator not running."

        self.max_episode_steps = 1000
        self.traj = []
        self.cumulative_reward = 0
        self.new_reward = True
        self.TARGET_OBJ_POSITION_KEY = (
            "sai2::ReinforcementLearning::push_puck_easy::puck_position"
        )
        self.GOAL_POSITION_KEY = (
            "sai2::ReinforcementLearning::push_puck_easy::goal_position"
        )
        self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"
        self.DESIRED_POS_KEY = "sai2::ReinforcementLearning::desired_position"
        # scaling factor for reaching reward
        self.cr = 1
        # scaling factor for pushing reward
        self.cp = 1
        self.lambda1 = 0
        self.lambda2 = 0
        self.reach_reward = []
        self.push_reward = []
        self.finished_reward = []

        if simulation:
            self.goal_position = self._client.redis2array(
                self._client.get(self.GOAL_POSITION_KEY)
            )

            self.current_puck_position = self.get_puck_position()
            self.last_obj_position = self.current_puck_position
            self.total_distance = self.euclidean_distance(
                self.goal_position, self.current_puck_position
            )

        else:
            # setup the things that we need in the real world
            # self.goal_position = None
            # self.current_puck_position = None
            # self.last_obj_position = None

            # new modify
            self.current_obj_distance = self.camera_handler.grab_distance()
            self.last_obj_distance = self.current_obj_distance
            self.total_distance = self.camera_handler.grab_distance()

    def initialize_task(self):
        self.cumulative_reward = 0
        # self.print_reward_statistics()
        self.reach_reward = []
        self.push_reward = []
        self.finished_reward = []
        if self._simulation:
            self.goal_position = self._client.redis2array(
                self._client.get(self.GOAL_POSITION_KEY)
            )
            self.current_puck_position = self.get_puck_position()
            self.last_obj_position = self.current_puck_position
            self.total_distance = (
                self.euclidean_distance(
                    self.goal_position, self.current_puck_position
                )
                - 0.04
            )
            # scale lambdas for the given configuration
            self.calculate_reward_scaling()
            # plan "optimal" trajectory
            self.traj = self.plan_optimal_trajectory()
        else:
            self.total_distance = self.camera_handler.grab_distance()

    def calculate_reward_scaling(self):
        self.current_puck_position = self.get_puck_position()
        direction = (
            self.current_puck_position[:2] - self.goal_position[:2]
        ) / self.euclidean_distance(
            self.current_puck_position, self.goal_position
        )
        reach_pos = self.current_puck_position[:2] + 0.05 * direction
        reach_pos = np.append(reach_pos, [0.05])
        self.lambda1 = 4 / self.euclidean_distance(
            self.get_ee_position(), reach_pos
        )
        self.lambda2 = 4 / self.euclidean_distance(
            self.current_puck_position[:2], self.goal_position[:2]
        )

    def print_reward_statistics(self):
        if self.reach_reward and self.push_reward and self.finished_reward:
            reached = np.sum(self.reach_reward)
            pushed = np.sum(self.push_reward)
            finished = np.sum(self.finished_reward)
            print(
                "Total: {:.2f}  | Reached: {:.2f} | Push: {:.2f} | Finished: {:.2f}".format(
                    reached + pushed + finished, reached, pushed, finished
                )
            )
            mreached = np.mean(self.reach_reward)
            mpushed = np.mean(self.push_reward)
            mfinished = np.mean(self.finished_reward)
            print(
                "Mean: {:.2f} | Reached: {:.2f} | Push: {:.2f} | Finished: {:.2f}".format(
                    mreached + mpushed + mfinished,
                    mreached,
                    mpushed,
                    mfinished,
                )
            )
            print(
                "Min/Max:    | Reached: {:.4f}/{:.4f} | Push: {:.4f}/{:.4f}".format(
                    np.min(self.reach_reward),
                    np.max(self.reach_reward),
                    np.min(self.push_reward),
                    np.max(self.push_reward),
                )
            )

    def compute_reward(self):
        """
        New 2 stage reward to guide policy
        1. stage: reach position behind the puck
        2. stage: push puck into the goal
        """
        goal_pos = self.goal_position
        puck_pos = self.get_puck_position()
        ee_pos = self.get_ee_position()

        # check if we are in stage 1 or 2
        direction = (puck_pos[:2] - goal_pos[:2]) / self.euclidean_distance(
            puck_pos, goal_pos
        )
        reach_pos = puck_pos[:2] + 0.05 * direction
        reach_pos = np.append(reach_pos, [0.05])
        reached = (ee_pos[0] - reach_pos[0]) ** 2 + (
            ee_pos[1] - reach_pos[1]
        ) ** 2 + (ee_pos[2] - reach_pos[2]) ** 2 <= 0.06 ** 2

        # print("Not reached", end="\r")
        reach_reward = self.cr * (
            1
            - (
                np.tanh(
                    self.lambda1 * self.euclidean_distance(reach_pos, ee_pos)
                )
            )
        )
        self.reach_reward.append(reach_reward)
        push_reward = self.cp * (
            1
            - (
                np.tanh(
                    self.lambda2
                    * self.euclidean_distance(goal_pos[:2], puck_pos[:2])
                )
            )
        )
        self.push_reward.append(push_reward)

        finish_reward = 1 if self.is_in_goal(puck_pos) else 0
        self.finished_reward.append(finish_reward)

        reward = reach_reward + push_reward + finish_reward
        done = self.is_in_goal(puck_pos)
        return reward, done

    def act_optimally(self):
        # only works for the moving target action space right now
        desired_pos = self.get_desired_position()
        ee_pos = self.get_ee_position()
        action = np.array([0, 0, 0, 0, 0])
        if self.traj:
            required_behavior = self.traj[0]
            required_position = required_behavior[:3]
            required_stiffness = required_behavior[3:]
            if self.euclidean_distance(required_position, ee_pos) > 0.02:
                action_pos = required_position - desired_pos[:3]
                # TODO add stiffness
                action = np.concatenate((action_pos, np.array([0, 0])))
            else:
                self.traj.pop(0)
        else:
            self.traj = self.plan_optimal_trajectory()

        return action

    def plan_optimal_trajectory(self):
        goal_pos = self.goal_position
        puck_pos = self.get_puck_position()
        direction = (puck_pos[:2] - goal_pos[:2]) / self.euclidean_distance(
            puck_pos, goal_pos
        )
        reach_pos = puck_pos[:2] + 0.13 * direction
        reach_pos = np.append(reach_pos, [0.05])
        # first action behind the
        a1 = np.array([reach_pos[0], reach_pos[1], 0.15, 50, 0])
        # # go down z direction
        a2 = np.array([reach_pos[0], reach_pos[1], 0.05, 50, 0])
        # # go to middle of the workspace
        # a3 = np.array([puck_pos[0], np.sign(puck_pos[1])*0.05, 0.05, 0, 0])
        # # go up again
        # a4 = np.array([puck_pos[0], np.sign(puck_pos[1])*0.05, 0.18, 0, 0])
        # # go behind puck again
        # a5 = np.array([puck_pos[0]-0.10, 0, 0.18, 0, 0])
        # # go down z again
        # a6 = np.array([puck_pos[0]-0.10, 0, 0.05, 0, 0])
        # push towards goal in (0.6,0,0)
        a7 = np.array(
            [
                self.goal_position[0],
                self.goal_position[1] + np.sign(puck_pos[1]) * 0.05,
                0.05,
                0,
                0,
            ]
        )

        # trajectory = [a1, a2, a3, a4, a5, a6, a7]
        trajectory = [a1, a2, a7]

        return trajectory

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def is_in_goal(self, pos):
        return (pos[0] - self.goal_position[0]) ** 2 + (
            pos[1] - self.goal_position[1]
        ) ** 2 <= 0.04 ** 2

    def get_ee_position(self):
        return self._client.redis2array(self._client.get(self.CURRENT_POS_KEY))

    def get_puck_position(self):
        return self._client.redis2array(
            self._client.get(self.TARGET_OBJ_POSITION_KEY)
        )

    def get_desired_position(self):
        return self._client.redis2array(self._client.get(self.DESIRED_POS_KEY))

    def compute_old_reward(self):
        """
        There is a total of 10 reward per episde.
        1 for pushing the object to the goal and 9 for completing the task.
        Reward is normalized by the initial distance.
        """
        # if self.new_reward:
        # return self.compute_new_reward()
        done = False
        reward = 0
        if self._simulation:
            self.last_obj_position = self.current_puck_position
            self.current_puck_position = self.get_puck_position()
            d0 = (
                self.euclidean_distance(
                    self.goal_position, self.last_obj_position
                )
                - 0.04
            )
            d1 = (
                self.euclidean_distance(
                    self.goal_position, self.current_puck_position
                )
                - 0.04
            )

            reward = (d0 - d1) / self.total_distance
            done = self.is_in_goal(self.current_puck_position)

        else:
            # reward = 0
            # TODO
            # new modify
            # When detecting no enough markers at the very beginning
            self.last_obj_distance = self.current_obj_distance
            self.current_obj_distance = self.camera_handler.grab_distance()
            d_last = self.last_obj_distance
            d_current = self.current_obj_distance
            if d_current == 1:
                reward = 0
            else:
                reward = (d_last - d_current) / self.total_distance
                # reward = d_current
            done = d_current < 0.04

        self.cumulative_reward += reward
        if done:
            reward += 1 - self.cumulative_reward
            reward += 9
        return reward, done

