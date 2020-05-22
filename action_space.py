from enum import Enum
import numpy as np
from gym.spaces import Box


class ActionSpace(Enum):
    NONE = 0
    ABS_JOINT_POSITION_DYN_DECOUP = 1
    DELTA_JOINT_POSITION_DYN_DECOUP = 2
    ABS_JOINT_POSITION_IMPEDANCE = 3
    DELTA_JOINT_POSITION_IMPEDANCE = 4

    ABS_JOINT_TORQUE = 5
    DELTA_JOINT_TORQUE = 6

    ABS_EE_POSE_DYN_DECOUP = 7
    DELTA_EE_POSE_DYN_DECOUP = 8
    ABS_EE_POSE_IMPEDANCE = 9
    DELTA_EE_POSE_IMPEDANCE = 10


class RobotAction(object):
    def __init__(self, action_space=None, isotropic_gains=True):
        self.action_space = action_space
        self.isotropic_gains = isotropic_gains
        #TODO will need to refine these min and max values
        #https://frankaemika.github.io/docs/control_parameters.html
        self._min_joint_position = np.array(
            [-2.7, -1.6, -2.7, -3.0, -2.7, 0.2, -2.7])
        self._max_joint_position = np.array(
            [2.7, 1.6, 2.7, -0.2, 2.7, 3.6, 2.7])
        self._min_joint_position_delta = np.full((7, ), -0.1)
        self._max_joint_position_delta = -1 * self._min_joint_position_delta

        self._max_joint_torques = np.array([85, 85, 85, 85, 10, 10, 10])
        self._max_joint_velocity = np.array(
            [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
        #TODO cartesian min max? not a box though

        max_kp = 250
        max_stiffness = 250
        iso_dim = 1 if isotropic_gains else 6

        self._min_kp = np.zeros((iso_dim, ))
        self._max_kp = np.full((iso_dim, ), max_kp)
        self._min_kp_delta = -1 * self._max_kp / 100
        self._max_kp_delta = -1 * self._min_kp_delta

        self._min_stiffness = np.zeros((iso_dim, ))
        self._max_stiffness = np.full((iso_dim, ), max_stiffness)
        self._min_stiffness_delta = -1 * self._max_stiffness / 100
        self._max_stiffness_delta = -1 * self._min_stiffness_delta

        self.dimensions = {
            ActionSpace.NONE:
            None,
            ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP:
            Box(low=np.concatenate((self._min_joint_position, self._min_kp)),
                high=np.concatenate((self._max_joint_position, self._max_kp)),
                dtype=np.float32),
            ActionSpace.DELTA_JOINT_POSITION_DYN_DECOUP:
            Box(low=np.concatenate(
                (self._min_joint_position_delta, self._min_kp_delta)),
                high=np.concatenate(
                    (self._max_joint_position_delta, self._max_kp_delta)),
                dtype=np.float32),
            ActionSpace.ABS_JOINT_POSITION_IMPEDANCE:
            Box(low=np.concatenate(
                (self._min_joint_position, self._min_stiffness)),
                high=np.concatenate(
                    (self._max_joint_position, self._max_stiffness)),
                dtype=np.float32),
            ActionSpace.DELTA_JOINT_POSITION_IMPEDANCE:
            Box(low=np.concatenate(
                (self._min_joint_position_delta, self._min_stiffness_delta)),
                high=np.concatenate((self._max_joint_position_delta,
                                     self._max_stiffness_delta)),
                dtype=np.float32),
            ActionSpace.ABS_JOINT_TORQUE:
            Box(low=np.zeros((7, )),
                high=self._max_joint_torques,
                dtype=np.float32),
            ActionSpace.DELTA_JOINT_TORQUE:
            Box(low=-0.1 * np.ones((7, )),
                high=0.1 * np.zeros((7, )),
                dtype=np.float32),
            ActionSpace.ABS_EE_POSE_DYN_DECOUP:
            Box(low=np.concatenate((np.full((7, ), -np.inf), self._min_kp)),
                high=np.concatenate((np.full((7, ), -np.inf), self._max_kp)),
                dtype=np.float32),
            ActionSpace.DELTA_EE_POSE_DYN_DECOUP:
            Box(low=np.concatenate((np.full((7, ),
                                            -np.inf), self._min_kp_delta)),
                high=np.concatenate((np.full((7, ),
                                             -np.inf), self._max_kp_delta)),
                dtype=np.float32),
            ActionSpace.ABS_EE_POSE_IMPEDANCE:
            Box(low=np.concatenate((np.full((7, ),
                                            -np.inf), self._min_stiffness)),
                high=np.concatenate((np.full((7, ),
                                             -np.inf), self._max_stiffness)),
                dtype=np.float32),
            ActionSpace.DELTA_EE_POSE_IMPEDANCE:
            Box(low=np.concatenate((np.full(
                (7, ), -np.inf), self._min_stiffness_delta)),
                high=np.concatenate((np.full(
                    (7, ), -np.inf), self._max_stiffness_delta)),
                dtype=np.float32),
        }

    def action_space_size(self):
        return self.dimensions[self.action_space]


def action_dimension(space):

    return None

