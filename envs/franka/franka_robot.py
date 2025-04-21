from .tora import TORA

from typing import Optional
import time
import numpy as np
from gymnasium import spaces

from .core import PyBulletRobot
from panda_gym.pybullet import PyBullet

# real franka panda robot
from scipy.spatial.transform import Rotation
from franky import Robot, Gripper
from franky import Affine, JointWaypointMotion, JointWaypoint, JointState, CartesianMotion, ReferenceType, CartesianWaypointMotion, CartesianWaypoint


class FrankaPanda(PyBulletRobot):
    """Real FrankaPanda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        robot_ip: str = '192.168.31.11', # left arm: '192.168.31.11'; right arm = '192.168.31.12'
        relative_dynamics_factor: float = 0.05,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = False
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )
        ########  Initialize the real Franka robot
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)

        self.tora = TORA()

        # Set velocity, acceleration and jerk to 5% of the maximum
        self.robot.relative_dynamics_factor = relative_dynamics_factor 
        self.robot.set_joint_impedance([50.]*7)

        # Alternatively, you can define each constraint individually
        # self.robot.velocity_rel = 0.1
        self.robot.acceleration_rel = 1.0
        self.robot.jerk_rel = 1.0

        self.relative_df = relative_dynamics_factor

        self.pose_shift = [0, 0, 0]     ###  a translation offset applied to the EE position
        self.start_joint_pose = []
        self.gripping_succuss=False

        # Sets the impedance for each joint in the internal controller.
        imp_value = 2000
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

        self.robot.recover_from_errors()

        self.prev_ee_position = None
        self.prev_time = None

        ######## the PyBulletRobot
        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray, asynchronous=False) -> None:
        """Update[2025/04/10]. Set the action of the robot."""
        action = action.copy()  # ensure action don't change
    
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            # ### set z offset while grasping
            # if action[-1] < 0.0:
            #     ### offset 1: robot会在刚刚卡住cube一点点的位置（z=0.06）开始lift，所以需要再real中加一个z方向向下的偏置
            #     ee_trans, ee_quat, ee_rpy = self.get_ee_position()
            #     trans = ee_trans + np.array([0.0, 0.0, -0.01])   # limit maximum change in position
            #     quat = ee_quat
            #     self.set_ee_pose(trans, quat, asynchronous=asynchronous)
            #     ee_trans, ee_quat, ee_rpy = self.get_ee_position()
            #     print("offset ee position:", ee_trans)

            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            print("open figger:", fingers_ctrl)
            # if fingers_ctrl > 0.0:
            #     fingers_ctrl += 0.005
            fingers_width = self.get_fingers_width()
            print("fingers_width:", fingers_width)
            target_fingers_width = fingers_width + fingers_ctrl
            self.set_gripper_opening(target_fingers_width, asynchronous=asynchronous)
            print("change fingerswidth:", self.get_fingers_width())

        if self.control_type == "ee":
            # method 1: set ee pose all together
            ee_displacement = action[:3]
            ee_trans, ee_quat, ee_rpy = self.get_ee_position()
            print("ee_rpy:", ee_rpy)
            # trans = ee_trans + ee_displacement[:3]
            trans = ee_trans + ee_displacement[:3] * 0.05  # limit maximum change in position
            quat = ee_quat
            self.set_ee_pose(trans, quat, asynchronous=asynchronous)
            # self.set_sora_pose(trans, quat, asynchronous=asynchronous)
            print("action after clip:", ee_displacement[:3] * 0.05, fingers_ctrl)
        else:
            raise NotImplementedError("Joint control is not implemented yet.")
        
        if self.gripping_succuss:
            ee_trans, end_ee_quat, ee_rpy = self.get_ee_position()
            end_action = np.array([0.6,0.02,0.07])
            self.set_ee_pose(end_action, end_ee_quat, asynchronous=asynchronous)
            self.gripping_succuss = False
            self.set_gripper_opening(0.08, asynchronous=asynchronous)
            self.reset()
            raise NotImplementedError("Gripping succuss, reset the robot.")


    def compute_ee_velocity_from_position(self, ee_position):
        current_time = time.time()
        ee_position = np.array(ee_position)

        if self.prev_ee_position is None:
            self.prev_ee_position = ee_position
            self.prev_time = current_time
            return np.zeros(3)

        dt = current_time - self.prev_time
        if dt == 0:
            return np.zeros(3)

        velocity = (ee_position - self.prev_ee_position) / dt

        self.prev_ee_position = ee_position
        self.prev_time = current_time
        return velocity

    def get_obs(self) -> np.ndarray:
        """Update[2025/04/10]. Get the observation of the robot."""
        # end-effector position and velocity
        ee_position, _, _ = self.get_ee_position()
        ee_position = np.array(ee_position)
        # ee_velocity = np.array(self.get_ee_velocity())
        ee_velocity =  self.compute_ee_velocity_from_position(ee_position)
        ee_velocity = np.array(ee_velocity)
        print("ee_position:", ee_position)
        # fingers opening
        # if not self.block_gripper:
        #     fingers_width = self.get_fingers_width()
        #     observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        # else:
        #     observation = np.concatenate((ee_position, ee_velocity))
        
        # [update 2025/04/10] delete joint velocity in the observation
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, [fingers_width]))
        else:
            observation = ee_position
        return observation

    def get_joint_pose(self):
        state = self.robot.state
        joint_pose = state.q
        return joint_pose
    
    def get_elbow_pose(self):
        state = self.robot.state
        elbow_pose = state.elbow
        return elbow_pose
    
    def get_joint_velocity(self):
        return self.robot.current_joint_velocities

    def set_ee_pose(self, translation, quaternion, asynchronous=True):
        shifted_translation = translation - self.pose_shift
        # if shifted_translation[-1] < 0.02:
        #     shifted_translation[-1] = 0.02
        motion = CartesianMotion(Affine(shifted_translation, quaternion))
        self.robot.move(motion, asynchronous=asynchronous)

    def set_sora_pose(self, ee_trans, ee_quat, asynchronous=False):
        target_ee_trans = ee_trans - self.pose_shift
        # if target_ee_trans[-1] < 0.02:
        #     target_ee_trans[-1] = 0.02
        target_ee_quat = ee_quat
        current_ee_trans = self.robot.current_pose.end_effector_pose.translation
        current_ee_quat = self.robot.current_pose.end_effector_pose.quaternion
        joint_pose = self.robot.state.q
        joint_v = self.robot.current_joint_velocities

        target_joint, _, _ = self.tora.plan(joint_pose, joint_v, current_ee_trans, current_ee_quat, target_ee_trans, target_ee_quat)
        # return target_joint
        self.set_joint_pose(target_joint[-1], asynchronous=asynchronous)

    def set_joint_pose(self, joint_pose, asynchronous=False):
        assert len(joint_pose) == 7
        m1 = JointWaypointMotion([JointWaypoint(joint_pose)])
        self.robot.move(m1, asynchronous=asynchronous)

    def set_gripper_opening(self, width, reset_flag = False, asynchronous=False):
        """commanding the gripper to move its fingers to a specific width (width), at a specified speed (0.03 m/s, or 3 cm/s)."""
        print("set gripper width:", width)
        ### offset 2: 避免发生碰撞
        if self.gripping_succuss:
            print("$$$$$$$$$$$$$$$$ moving to goal")
            return
        if width < 0.060 and not reset_flag and not self.gripping_succuss:
            width = 0.064
            self.gripping_succuss = True
            print("$$$$$$$$$$$$$$$$ gripping succuss")

        if asynchronous:
            self.gripper.move_async(width, 0.03)
        else:
            self.gripper.move(width, 0.03)

    def reset(self) -> None:
        """Update[2025/04/10]. Reset the robot to its default state."""
        self.set_joint_pose(self.neutral_joint_values[:7])
        self.set_gripper_opening(0.0, True) # set gripper to closed

    def get_fingers_width(self) -> float:
        """Update[2025/04/10]. Get the distance between the fingers."""
        return self.gripper.width
    
    def get_ee_position(self) -> np.ndarray:
        """Update[2025/04/10].Returns the position of the end-effector as (x, y, z)"""
        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion ### A quaternion representing the orientation of the end-effector. Format: [x, y, z, w].
        ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz') # Converts that rotation into Euler angles: [roll (around x), pitch (around y), yaw (around z)]

        shifted_ee_trans = ee_trans + self.pose_shift
        # return self.get_link_position(self.ee_link)
        return shifted_ee_trans, ee_quat, ee_rpy
        # return shifted_ee_trans

    def get_ee_velocity(self) -> np.ndarray:
        """Update[2025/04/10].Returns the velocity of the end-effector as (vx, vy, vz)"""
        twist = self.robot.current_cartesian_velocity.end_effector_twist  # type: CartesianTwist
        linear_velocity = twist.linear      # np.ndarray of shape (3, 1)
        angular_velocity = twist.angular    # np.ndarray of shape (3, 1)
        ee_velocity = np.array(linear_velocity)

        # return self.get_link_velocity(self.ee_link)
        return ee_velocity