import math
from time import sleep
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from franky import Robot, Gripper
from franky import Affine, JointWaypointMotion, JointWaypoint, JointState, CartesianMotion, ReferenceType, CartesianWaypointMotion, CartesianWaypoint


class Franka:
    def __init__(self, robot_ip, relative_dynamics_factor=0.05) -> None:
        self.robot = Robot(robot_ip)
        self.gripper = Gripper(robot_ip)

        self.robot.relative_dynamics_factor = relative_dynamics_factor
        # self.robot.velocity_rel = 0.1
        self.robot.acceleration_rel = 1.0
        self.robot.jerk_rel = 1.0

        self.relative_df = relative_dynamics_factor

        self.pose_shift = [0, 0, 0]
        self.start_joint_pose = []
        imp_value = 2000
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

        self.robot.recover_from_errors()
        
    def set_soft(self, imp_value=50):
        imp_value = imp_value
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def set_hard(self):
        imp_value = 1000
        self.robot.set_joint_impedance([imp_value, imp_value, imp_value, imp_value, imp_value, imp_value, imp_value])

    def speed_down(self):
        self.robot.relative_dynamics_factor = self.relative_df * 0.5

    def speed_normal(self):
        self.robot.relative_dynamics_factor = self.relative_df

    def set_default_pose(self):
        self.robot.relative_dynamics_factor = 0.07
        motion = CartesianMotion(Affine([-0.05, 0.0, 0.0]), ReferenceType.Relative)
        self.robot.move(motion)
        self.set_joint_pose(self.start_joint_pose, asynchronous=False)
        self.robot.relative_dynamics_factor = self.relative_df

    def open_gripper(self, asynchronous=False):
        # if asynchronous:
        #     self.gripper.move_async(0.04, 0.02)
        # else:
        #     self.gripper.open(0.04)
        # self.gripper.move_async(0.04, 0.03)    
        success = self.gripper.move(0.04, 0.03)

    def close_gripper(self, asynchronous=False):
        # if asynchronous:
        #     self.gripper.move_async(0.0, 0.03)
        # else:
        #     self.gripper.move(0.0, 0.03)
        force = 20
        # self.gripper.move_async(0.0, 0.03)
        # gripper.grasp(0.0, speed, force, epsilon_outer=1.0)

        # self.gripper.grasp(0.0, 0.03, force, epsilon_outer=1.0);        
        success = self.gripper.move(0.01, 0.03)

    def set_gripper_opening(self, width, asynchronous=False):
        if asynchronous:
            self.gripper.move_async(width, 0.03)
        else:
            self.gripper.move(width, 0.03)

    def set_ee_pose_relative(self, translation, asynchronous=False):
        motion = CartesianMotion(Affine(translation), ReferenceType.Relative)
        self.robot.move(motion, asynchronous=asynchronous)

    def set_ee_pose(self, translation, quaternion, asynchronous=True):
        shifted_translation = translation - self.pose_shift
        motion = CartesianMotion(Affine(shifted_translation, quaternion))
        self.robot.move(motion, asynchronous=asynchronous)

    def set_joint_pose(self, joint_pose, asynchronous=False):
        assert len(joint_pose) == 7
        m1 = JointWaypointMotion([JointWaypoint(joint_pose)])
        self.robot.move(m1, asynchronous=asynchronous)

    def get_ee_pose(self):
        robot_pose = self.robot.current_pose
        ee_trans = robot_pose.end_effector_pose.translation
        ee_quat = robot_pose.end_effector_pose.quaternion
        ee_rpy = Rotation.from_quat(ee_quat).as_euler('xyz')

        shifted_ee_trans = ee_trans + self.pose_shift
        return shifted_ee_trans, ee_quat, ee_rpy

    def get_joint_pose(self):
        state = self.robot.state
        joint_pose = state.q
        return joint_pose

    def get_elbow_pose(self):
        state = self.robot.state
        elbow_pose = state.elbow
        return elbow_pose
    
    def get_joint_vel(self):
        return self.robot.current_joint_velocities
    
    def get_gripper_width(self):
        return self.gripper.width