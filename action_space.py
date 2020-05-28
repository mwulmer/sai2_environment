from enum import Enum
import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation as Rot
from ipdb import set_trace


class ActionSpace(Enum):
    NONE = 0
    """
    Digits Notation:
    | Space | Abs/Delta | Dyn decoup/ impedance |
    """
    #anisotropic: joints (7) Kp/stiffness (7) 
    #isotropic: joints (7) Kp/stiffness (1) 
    ABS_JOINT_POSITION_DYN_DECOUP = 111
    DELTA_JOINT_POSITION_DYN_DECOUP = 121
    ABS_JOINT_POSITION_IMPEDANCE =  112
    DELTA_JOINT_POSITION_IMPEDANCE = 122

    #joints (7)
    ABS_JOINT_TORQUE = 110    
    DELTA_JOINT_TORQUE = 120

    #anisotropic: position (3) rotation (4) Kp/stiffness: translational (3) rotational (3)
    #isotropic: position (3) rotation (4) Kp/stiffness: translational (1) rotational (1)
    ABS_EE_POSE_DYN_DECOUP = 211
    DELTA_EE_POSE_DYN_DECOUP = 221 
    ABS_EE_POSE_IMPEDANCE = 212 
    DELTA_EE_POSE_IMPEDANCE = 222 

class RobotAction(object):
    def __init__(self, action_space_enum, isotropic_gains=True):
        self.action_space_enum = action_space_enum
        self.isotropic_gains = isotropic_gains

    def decode_action_space(self, action_space):
        i = action_space.value
        return i // 10**2 % 10, i // 10**1 % 10, i // 10**0 % 10

    def action_space_size(self):
        raise NotImplementedError()    

    def build_full_command(self, action):
        raise NotImplementedError()

class JointSpaceAction(RobotAction):
    def __init__(self, action_space_enum, isotropic_gains=True):
        super().__init__(action_space_enum, isotropic_gains=isotropic_gains)
        self.space_type, self.value_type, self.controller_type = self.decode_action_space(self.action_space_enum)

        #action space is pure torques
        if self.controller_type == 0:
            self.action_space = Box(low=np.zeros((7, )), high=self._max_joint_torques, dtype=np.float32)





class RobotAction(object):
    def __init__(self, action_space=None, isotropic_gains=True, rotation_axis=(True, True, True)):
        self.action_space = action_space
        #decode the number for later use
        self.space_type, self.value_type, self.controller_type = self.decode_action_space(self.action_space)
        self.isotropic_gains = isotropic_gains
        self.rotation_axis = rotation_axis
        
        # TODO will need to refine these min and max values
        # https://frankaemika.github.io/docs/control_parameters.html
        self._min_joint_position = np.array([-2.7, -1.6, -2.7, -3.0, -2.7, 0.2, -2.7])
        self._max_joint_position = np.array([2.7, 1.6, 2.7, -0.2, 2.7, 3.6, 2.7])
        self._min_joint_position_delta = np.full((7, ), -0.1)
        self._max_joint_position_delta = -1 * self._min_joint_position_delta

        self._max_joint_torques = np.array([85, 85, 85, 85, 10, 10, 10])
        self._max_joint_velocity = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
        # TODO cartesian min max? not a box though

        max_kp = 250
        max_stiffness = 250
        #determine the dimension of the gain vector
        if self.space_type == 1:
            #one value per join
            #isotropic gains probably dont make a lot of sense here
            self.gains_dim = 1 if isotropic_gains else 7
        else:
            #one value for each direction translational and rotational            
            self.gains_dim = 2 if isotropic_gains else 6

        #TODO min gains cant be 0 
        self._min_kp = np.zeros((self.gains_dim, ))
        self._max_kp = np.full((self.gains_dim, ), max_kp)
        self._min_kp_delta = -1 * self._max_kp / 100
        self._max_kp_delta = -1 * self._min_kp_delta

        self._min_stiffness = np.zeros((self.gains_dim, ))
        self._max_stiffness = np.full((self.gains_dim, ), max_stiffness)
        self._min_stiffness_delta = -1 * self._max_stiffness / 100
        self._max_stiffness_delta = -1 * self._min_stiffness_delta

        pose_dim = 7 if sum(rotation_axis) == 3 else 7 - 4 + sum(rotation_axis)

        self.dimensions = {
            ActionSpace.NONE:
            None,
            ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP:
            Box(low=np.concatenate((self._min_joint_position, self._min_kp)),
                high=np.concatenate((self._max_joint_position, self._max_kp)),
                dtype=np.float32),
            ActionSpace.DELTA_JOINT_POSITION_DYN_DECOUP:
            Box(low=np.concatenate((self._min_joint_position_delta, self._min_kp_delta)),
                high=np.concatenate((self._max_joint_position_delta, self._max_kp_delta)),
                dtype=np.float32),
            ActionSpace.ABS_JOINT_POSITION_IMPEDANCE:
            Box(low=np.concatenate((self._min_joint_position, self._min_stiffness)),
                high=np.concatenate((self._max_joint_position, self._max_stiffness)),
                dtype=np.float32),
            ActionSpace.DELTA_JOINT_POSITION_IMPEDANCE:
            Box(low=np.concatenate((self._min_joint_position_delta, self._min_stiffness_delta)),
                high=np.concatenate((self._max_joint_position_delta, self._max_stiffness_delta)),
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
            Box(low=np.concatenate((np.full((pose_dim, ), -np.inf), self._min_kp)),
                high=np.concatenate((np.full((pose_dim, ), -np.inf), self._max_kp)),
                dtype=np.float32),
            ActionSpace.DELTA_EE_POSE_DYN_DECOUP:
            Box(low=np.concatenate((np.full((pose_dim, ), -np.inf), self._min_kp_delta)),
                high=np.concatenate((np.full((pose_dim, ), -np.inf), self._max_kp_delta)),
                dtype=np.float32),
            ActionSpace.ABS_EE_POSE_IMPEDANCE:
            Box(low=np.concatenate((np.full((pose_dim, ), -np.inf), self._min_stiffness)),
                high=np.concatenate((np.full((pose_dim, ), -np.inf), self._max_stiffness)),
                dtype=np.float32),
            ActionSpace.DELTA_EE_POSE_IMPEDANCE:
            Box(low=np.concatenate((np.full((pose_dim, ), -0.1), np.full(self.gains_dim,400))),#self._min_stiffness_delta)),
                high=np.concatenate((np.full((pose_dim, ), 0.1), np.full(self.gains_dim,500))),#self._max_stiffness_delta)),
                dtype=np.float32),
        }

    def action_space_size(self):
        return self.dimensions[self.action_space]

    def decode_action_space(self, action_space):
        i = action_space.value
        return i // 10**2 % 10, i // 10**1 % 10, i // 10**0 % 10

    def rotvec_to_quaternion(self, vec):
        quat = Rot.from_euler('zyx', vec).as_quat()
        #[w, x, y, z]
        idx = [3, 0, 1, 2] 
        return quat[idx] 

    def build_full_command(self, action):
        #this is needed if we use isotropic gains or limit the rotation of the EE
        #if directly we send torques, or the policy generates the full command (-> no isotropic gains and full quaternion rotation)

        if self.controller_type == 0 or (not self.isotropic_gains and sum(self.rotation_axis)==3):
            return action
        
        #if we use isotropic gains with a joint space
        if self.space_type == 1 and self.isotropic_gains:
            q = action[:7]
            kp = np.full((7, ), action[7])
            return np.concatenate((q,kp))

        #build the quationions if we dont have full rotation
        if sum(self.rotation_axis) != 3:
            rotation_vector = action[3:3+sum(self.rotation_axis)]
        

        return None



def action_dimension(space):
    return None

