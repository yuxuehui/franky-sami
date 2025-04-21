from envs.franka.franka_robot import FrankaPanda
import numpy as np
from panda_gym.pybullet import PyBullet

left_robot_ip='192.168.31.11'
# right_robot_ip='192.168.31.12'

sim = PyBullet(render_mode="rgb_array", renderer="Tiny")
arm = FrankaPanda(sim, block_gripper=False, base_position=np.array([0.0, 0.0, 0.0]), control_type="ee", robot_ip=left_robot_ip)

# Current joint state
ee_trans, ee_quat, ee_rpy = arm.get_ee_position()
print(ee_trans, ee_rpy)

rl_action = np.array([
    0.0, 0.0, 0.0,  # ee displacement
    0.02,  # gripper opening
])

arm.set_action(rl_action, asynchronous=False)