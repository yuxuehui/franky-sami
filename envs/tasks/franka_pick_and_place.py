from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from abc import abstractmethod

import tf,cv2
import rospy, time
import signal
import sys, os
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        use_rgb: bool = False,
        arm_name: str = "left_arm",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.2, #0.3,
        goal_z_range: float = 0.2, # 0.2,
        obj_xy_range: float = 0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        # self.object_size = 0.04
        self.object_size = 0.06

        # # setting 1:
        # # goal space: [0.4, -0.1, 0.03] - [0.6, 0.1, 0.03]
        # # cube space: [0.4, -0.1, 0.03] - [0.6, 0.1, 0.03]
        # self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        # self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        # self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        # self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        # self.base_position_offset = np.array([0.5, 0.0, 0.0])

        # setting 2:
        # goal space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
        # cube space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
        self.goal_range_low = np.array([-goal_xy_range / 2, -1.5*goal_xy_range, 0])
        self.goal_range_high = np.array([goal_xy_range, 1.5*goal_xy_range, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -1.5*obj_xy_range, 0])
        self.obj_range_high = np.array([obj_xy_range, 1.5*obj_xy_range, 0])
        self.base_position_offset = np.array([0.5, 0.0, 0.0])

        with self.sim.no_rendering():
            self._create_scene()
        
        # franka cube data
        rospy.init_node("franka_cube_moving")
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.rate = rospy.Rate(10)  # 10 Hz loop rate
        self.prev_trans = None
        self.prev_rot = None
        self.prev_time = rospy.Time.now()
        self.cube_position_offset = np.array([0.01, -0.01, 0.015])
        # self.cube_position_offset = np.array([0.01, -0.01, 0.017])
        
        signal.signal(signal.SIGINT, self.signal_handler)

        self.use_rgb = use_rgb
        self.arm_name = arm_name  # "left" or "right"

        # camera init
        # building
        
        self.gripper_state = 0.05 # open

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        # path = data['path']
        # with open(path, 'wb') as f:
        #     pickle.dump(data, f)   
        sys.exit(0)

    def publish_static_tf(self, broadcaster):
        """Publish a static transform from franka_table to franka_base."""
        broadcaster.sendTransform(
            (-1.0+0.03, 0.0, 0.015),  # Translation
            (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
            rospy.Time.now(),
            "franka_base",  # Child frame
            "/vicon/franka_table/franka_table"  # Parent frame
        )
        broadcaster.sendTransform(
            (-0.007, 0.3125, 0),  # Translation
            (0.0, 0.0, 0.0, 1.0),  # Quaternion (no rotation)
            rospy.Time.now(),
            "franka_left",  # Child frame
            "franka_base"  # Parent frame
        )
    def is_valid_quaternion(q):
        q = np.array(q)
        return q.shape == (4,) and not np.isclose(np.linalg.norm(q), 0.0)


    def get_obs(self) -> np.ndarray:
        get_cube_obs_flag = False
        while not get_cube_obs_flag:
            self.publish_static_tf(self.broadcaster)
            current_time = rospy.Time.now()
            try:
                (cube_trans, cube_rot) = self.listener.lookupTransform('franka_left', 'vicon/franka_cube/franka_cube', rospy.Time(0))
                cube_ee_rpy = Rotation.from_quat(cube_rot).as_euler('xyz')
                # print("\n","cube", cube_trans, cube_ee_rpy*180/np.pi)
                get_cube_obs_flag = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform from franka_cube to franka_left, try again.")
                print("Failed to lookup transform from franka_cube to franka_left, try again.")

        object_position = cube_trans + self.cube_position_offset
        self.sim.set_base_pose("object", object_position, cube_rot)
        object_rotation = cube_rot
        object_velocity, object_angular_velocity = self.compute_velocity(object_position, object_rotation, current_time)
        # observation = np.concatenate([object_position, cube_ee_rpy, object_velocity, object_angular_velocity])
        observation = np.concatenate([object_position, cube_ee_rpy])
        if object_position[2] < 0.03:
            object_position[2] = 0.03
        # observation = np.concatenate([object_position, np.array([0.0, 0.0, 0.0])])
        return observation
    
    def compute_velocity(self, cube_trans, cube_rot, current_time):
        if self.prev_trans is None:
            self.prev_trans = cube_trans
            self.prev_rot = cube_rot
            self.prev_time = current_time 
            return np.zeros(3), np.zeros(3)
        # Time difference
        dt = (current_time - self.prev_time).to_sec() 
        # print("Time difference", dt)
        if dt == 0:
            return np.zeros(3), np.zeros(3)
        # Linear velocity
        trans1 = np.array(self.prev_trans)
        trans2 = np.array(cube_trans)
        linear_velocity = (trans2 - trans1) / dt
        # Angular velocity
        r1 = Rotation.from_quat(self.prev_rot)
        r2 = Rotation.from_quat(cube_rot)
        r_delta = r2 * r1.inv()
        rotvec = r_delta.as_rotvec()
        angular_velocity = rotvec / dt  # rad/s
        # Update previous state
        self.prev_trans = cube_trans
        self.prev_rot = cube_rot
        self.prev_time = current_time
        return linear_velocity, angular_velocity

    def get_achieved_goal(self) -> np.ndarray:
        get_cube_obs_flag = False
        while not get_cube_obs_flag:
            self.publish_static_tf(self.broadcaster)
            try:
                (cube_trans, cube_rot) = self.listener.lookupTransform('franka_left', 'vicon/franka_cube/franka_cube', rospy.Time(0))
                cube_ee_rpy = Rotation.from_quat(cube_rot).as_euler('xyz')
                get_cube_obs_flag = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform from franka_cube to franka_left, try again.")
                print("Failed to lookup transform from franka_cube to franka_left, try again.")
        object_position = cube_trans + self.cube_position_offset
        return np.array(object_position)

    def reset(self) -> None:
        self.goal = self._sample_goal() 
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.prev_trans = None
        self.prev_rot = None
        self.prev_time = rospy.Time.now()

    @abstractmethod
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        pass
    # def _sample_goal(self) -> np.ndarray:
    #     """Sample a goal."""
    #     goal = np.array([0.0, 0.0, self.object_size / 2]) + self.base_position_offset + self.cube_position_offset  # z offset for the cube center
    #     noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
    #     if self.np_random.random() < 0.3:
    #         noise[2] = 0.0
    #     goal += noise
    #     return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        get_cube_obs_flag = False
        while not get_cube_obs_flag:
            self.publish_static_tf(self.broadcaster)
            try:
                (cube_trans, cube_rot) = self.listener.lookupTransform('franka_left', 'vicon/franka_cube/franka_cube', rospy.Time(0))
                cube_ee_rpy = Rotation.from_quat(cube_rot).as_euler('xyz')
                # print("\n","cube", cube_trans, cube_ee_rpy*180/np.pi)
                get_cube_obs_flag = True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform from franka_cube to franka_left, try again.")
                print("Failed to lookup transform from franka_cube to franka_left, try again.")
        object_position = cube_trans + self.cube_position_offset
        object_position[2] = 0.03
        
        return np.array(object_position)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)