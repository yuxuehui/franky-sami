#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   PandaPush.py
@Author  :   Lixin & Xuehui 
@Version :   1.0
@Desc    :   None
'''

from .franka.franka_robot import FrankaPanda
from .franka.sim_panda import Panda

from typing import Any, Dict, Optional, Tuple

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from .tasks.pick_and_place import PickAndPlace

import gymnasium
from .PandaBase import PandaBaseWrapper,PandaBaseEnv

class PushWrapper(PandaBaseWrapper):
    
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  + self.base_position_offset + self.cube_position_offset # z offset for the cube center + base_position_offset
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        goal[2] = 0.03

        return np.array([0.6,0.02,0.03])
    

class PandaPushEnv(PandaBaseEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """
    
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 0.8,
        render_yaw: float = 0,
        render_pitch: float = 0,
        render_roll: float = 0,
        object_height: float = 1,
        causal_dim:int = -1,
        causal_hidden_dim:int = -1
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = FrankaPanda(sim, block_gripper=False, base_position=np.array([0.0, 0.0, 0.0]), control_type=control_type)
        task = PushWrapper(sim, reward_type=reward_type,object_height=object_height,goal_z_range=0.0)
        self.friction = 1.0
        self.mass = 1.0
        self.causal_dim = causal_dim
        self.causal_hidden_dim = causal_hidden_dim
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        if self.causal_hidden_dim > 0:
            # self.observation_space['friction'] = gymnasium.spaces.Box(-10,10,(1,),dtype=np.float32)
            # self.observation_space['mass'] = gymnasium.spaces.Box(-10,10,(1,),dtype=np.float32)
            self.observation_space['causal'] = gymnasium.spaces.Box(-10,10,(causal_dim,),dtype=np.float32)
            self.observation_space['hidden_h'] = gymnasium.spaces.Box(-10,10,(causal_hidden_dim,),dtype=np.float32)
            self.observation_space['hidden_c'] = gymnasium.spaces.Box(-10,10,(causal_hidden_dim,),dtype=np.float32)
