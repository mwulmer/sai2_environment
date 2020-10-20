import time
import cv2
import numpy as np
import pyrealsense2 as rs
from gym import spaces, core
from ipdb import set_trace
from scipy.spatial.transform import Rotation as Rot
from skimage.transform import resize

from sai2_environment.utils.client import RedisClient
from sai2_environment.utils.action_space import *
from sai2_environment.utils.misc import name_to_task_class, Timer
from sai2_environment.utils.ranges import Range, RobotMinMaxScaler
from sai2_environment.handlers.camera_handler import CameraHandler
from sai2_environment.handlers.haptic_handler import HapticHandler


class RobotEnv(core.Env):
    """
    The central wrapper around the robot control.
    """

    def __init__(
        self,
        domain_name="push_puck",
        task_name="easy",
        simulation=True,
        render=False,
        action_space=ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP,
        isotropic_gains=True,
        blocking_action=False,
        action_frequency=20,
        torque_seq_length=32,
        camera_available=True,
        camera_res=(128, 128),
        rotation_axis=(False, False, False),
        from_pixels=False,
    ):

        self.camera_available = camera_available
        self.full_name = domain_name + "_" + task_name
        self.from_pixels = from_pixels
        self.camera_resolution = camera_res
        # connect to redis server
        hostname = "127.0.0.1" if simulation else "TUEIRSI-RL-001"
        self.env_config = {
            "simulation": simulation,
            "render": render,
            "camera_resolution": camera_res,
            "camera_frequency": 30,
            "hostname": hostname,
            "port": 6379,
            "blocking_action": blocking_action,
            "rotation_axis": rotation_axis,
            "torque_seq_length": torque_seq_length,
        }

        # connect redis client
        self._client = RedisClient(config=self.env_config)
        self._client.connect()

        self.timer = Timer(frequency=action_frequency)
        self.start_time = time.time()

        # set action space to redis
        self._robot_action = get_robot_action(
            action_space, isotropic_gains, rotation_axis
        )

        self._client.init_action_space(self._robot_action)
        self._episodes = 0
        self._time_step = 0

        self._true_action_space = self._robot_action.action_space
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

        self.haptic_handler = (
            (
                HapticHandler.getInstance(
                    self._client, simulation, sensor_frequency=1000
                )
            )
            if not self.from_pixels
            else None
        )
        self.camera_handler = CameraHandler.getInstance(
            self.env_config["camera_resolution"]
        )

        self.scaler = RobotMinMaxScaler()

        self.current_frame = np.zeros(self.camera_resolution)
        if not self.env_config["simulation"] and self.camera_available:
            self.camera_handler.camera_thread.start()
        # ẃarm up camera

        if self.env_config["render"]:
            cv2.namedWindow("Simulator", cv2.WINDOW_NORMAL)

        time.sleep(1)

        self.observation_space = self.make_observation_space()

        # TODO define what all the responsibilites of task are
        task_class = name_to_task_class(self.full_name)
        self.task = task_class(
            "tmp",
            self._client,
            camera_handler=self.camera_handler,
            simulation=simulation,
        )
        self._max_episode_steps = self.task.max_episode_steps

    @property
    def action_space(self):
        return self._norm_action_space

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert (
            action.shape == self._robot_action.action_space_size()
        ), "Action shape of {} not correct, expected shape {}".format(
            action.shape, self._robot_action.action_space_size()
        )
        # build real action values from normed values
        action = self._convert_action(action)
        # build the full vector from the reduced values if
        action = self._robot_action.build_full_command(action)

        # blocking action waits until the action is carried out and computes reward along the trajectory
        if self.env_config["blocking_action"]:
            # first check if there is still something going on on the robot
            # print("Waiting for robot: {}".format(
            # self._client.action_complete()))
            self.take_action(action)
            time.sleep(0.01)

            while not self._client.action_complete():
                time.sleep(0.01)

            reward, done = self._compute_reward()

        # non-blocking does not wait and computes reward right away
        else:

            self.take_action(action)
            self.timer.wait_for_next_loop()

            reward, done = self._compute_reward()

        self._time_step += 1
        if self._time_step == self._max_episode_steps:
            done = True

        info = None
        obs = self._get_obs()  # has to be before the contact reset \!/

        return obs, reward, done, info

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        self._time_step = 0
        self._client.reset(self._episodes)
        # TODO do we want to set it every time or keep one action space per experiment?
        self._client.set_action_space()

        self._episodes += 1
        self.task.initialize_task()
        return self._get_obs()

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        """
        data = self.current_frame.copy()
        if mode == "rgb_array":
            return data
        elif mode == "human":
            cv2.imshow("Simulator", data)
            cv2.waitKey(1)
        else:
            print("only support rgb_array mode, given %s" % mode)

    def close(self):
        return 0

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

    def convert_image(self, im):
        return np.rollaxis(im, axis=2, start=0)

    def rotvec_to_quaternion(self, vec):
        quat = Rot.from_euler("zyx", vec).as_quat()
        # [w, x, y, z]
        idx = [3, 0, 1, 2]
        return quat[idx]

    def quaternion_to_rot(self, quaternion):
        return Rot.from_quat(quaternion).as_dcm()

    def take_action(self, action):
        return self._client.take_action(action)

    def act_optimally(self):
        action = self.task.act_optimally()
        return action

    def _compute_reward(self):
        reward, done = self.task.compute_reward()
        return reward, done

    def _get_obs(self):
        """
        camera_frame: im = (128,128)
        robot_state: (q,dq) = (14,)
        haptic_feedback: (tau, contact) = ((7,n), (1,))
        """
        output = dict()
        if self.env_config["simulation"]:
            img = self._client.get_camera_frame()
        else:
            img = (
                self.camera_handler.get_color_frame()
                if self.camera_available
                else 0
            )
        self.current_frame = img

        # resize to desired dimensions
        camera_frame = resize(
            img, (self.camera_resolution[0], self.camera_resolution[1]),
        )

        # rollaxis for tensor and make it unint8
        camera_frame = self.convert_image(camera_frame).astype(np.uint8)
        output["camera"] = camera_frame

        if self.from_pixels:
            return camera_frame
        else:
            # retrieve robot state
            q, dq = self._client.get_robot_state()
            # normalize proprioception
            q = self.scaler.q_scaler.transform([q])[0]
            dq = self.scaler.dq_scaler.transform([dq])[0]
            normalized_robot_state = np.concatenate((q, dq))
            output["proprioception"] = normalized_robot_state

            # retrieve haptics
            tau = self.haptic_handler.get_torques_matrix(
                n=self.env_config["torque_seq_length"]
            )
            contact = np.asarray([self.haptic_handler.contact_occured()])
            # normalize haptics
            tau = self.scaler.tau_scaler.transform(tau)
            reversed__transposed_tau = np.transpose(tau[::-1])

            normalized_haptic_feedback = (reversed__transposed_tau, contact)
            output["haptics"] = normalized_haptic_feedback

        return output["camera"], output["proprioception"], output["haptics"]

    def make_observation_space(self):
        height = self.camera_resolution[0]
        width = self.camera_resolution[1]
        # TODO what is the prefered way to do this with multiple different obs
        if self.from_pixels:
            shape = [3, height, width]
            observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            cam, proprio, haptic = self._get_obs()
            observation_space = {
                "camera": cam.shape,
                "proprioception": proprio.shape,
                "haptic": (haptic[0].shape, haptic[1].shape),
            }

        return observation_space

