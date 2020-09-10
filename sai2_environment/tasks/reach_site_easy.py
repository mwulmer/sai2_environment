from sai2_environment.tasks.task import Task
import numpy as np
import psutil


np.set_printoptions(precision=3, suppress=True)


class ReachSiteEasy(Task):
    def __init__(
        self, task_name, redis_client, camera_handler, simulation=True
    ):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.camera_handler = camera_handler

        # If simulated environment, check if simulation process is running
        if self._simulation:
            assert "sim02-reach_site_easy" in (
                p.name() for p in psutil.process_iter()
            ), "Simulator not running."

        self.max_episode_steps = 1000
        self.traj = []
        self.cumulative_reward = 0

        self.GOAL_POSITION_KEY = (
            "sai2::ReinforcementLearning::reach_site_easy::goal_position"
        )
        self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"
        self.DESIRED_POS_KEY = "sai2::ReinforcementLearning::desired_position"
        # scaling factor for reaching reward
        self.cr = 1
        self.lambda1 = None

        self.goal_position = None

        self.reach_reward = []
        self.finished_reward = []

    def initialize_task(self):
        self.cumulative_reward = 0
        # self.print_reward_statistics()
        self.reach_reward = []
        self.finished_reward = []
        if self._simulation:
            self.goal_position = self._client.redis2array(
                self._client.get(self.GOAL_POSITION_KEY)
            )
            # scale lambdas for the given configuration
            self.calculate_reward_scaling()
            # plan "optimal" trajectory
            self.traj = self.plan_optimal_trajectory()

    def calculate_reward_scaling(self):
        # substract the radius of the goal sphere
        self.lambda1 = 3 / (
            self.euclidean_distance(self.get_ee_position(), self.goal_position)
            - 0.1
        )

    def compute_reward(self):
        ee_pos = self.get_ee_position()

        # print("Not reached", end="\r")
        if not self.is_in_goal(ee_pos):
            reward = self.cr * (
                1
                - (
                    np.tanh(
                        self.lambda1
                        * (
                            self.euclidean_distance(self.goal_position, ee_pos)
                            - 0.1
                        )
                    )
                )
            )
            self.reach_reward.append(reward)
        else:
            reward = 1
            self.finished_reward.append(reward)

        return reward, False

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

        # first action go to the goal
        a1 = np.array([goal_pos[0], goal_pos[1], goal_pos[2], 50, 0])

        # trajectory = [a1, a2, a3, a4, a5, a6, a7]
        trajectory = [a1]

        return trajectory

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def is_in_goal(self, pos):
        return self.euclidean_distance(pos, self.goal_position) <= 0.1

    def get_ee_position(self):
        return self._client.redis2array(self._client.get(self.CURRENT_POS_KEY))

    def get_desired_position(self):
        return self._client.redis2array(self._client.get(self.DESIRED_POS_KEY))

    def print_reward_statistics(self):
        if self.reach_reward and self.finished_reward:
            reached = np.sum(self.reach_reward)
            finished = np.sum(self.finished_reward)
            print(
                "Total: {:.2f}  | Reached: {:.2f} | Finished: {:.2f}".format(
                    reached + finished, reached, finished
                )
            )
            mreached = np.mean(self.reach_reward)
            mfinished = np.mean(self.finished_reward)
            print(
                "Mean: {:.2f} | Reached: {:.2f} | Finished: {:.2f}".format(
                    mreached + mfinished, mreached, mfinished,
                )
            )
            print(
                "Min/Max:    | Reached: {:.4f}/{:.4f}".format(
                    np.min(self.reach_reward), np.max(self.reach_reward),
                )
            )

