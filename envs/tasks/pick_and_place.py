from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from abc import abstractmethod

class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.2,
        goal_z_range: float = 0.2,
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

        self.cube_position_offset = np.array([0.0, 0.0, 0.0])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """ _create_scene will be called in PandaBaseWrapper class
        Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.1, width=1.7, height=0.4, x_offset=1.0)
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

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object") # default "euler"
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        # observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        observation = np.concatenate([object_position, object_rotation])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    @abstractmethod
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        pass

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2 * self.object_height])  + self.base_position_offset + self.cube_position_offset
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)