import numpy as np
import cv2, os
import pyrealsense2 as rs
# from utils import ARUCO_DICT, aruco_display, get_center
from scipy.spatial.transform import Rotation as R

# Device: Intel RealSense D405, Serial Number: "419122270338"
# Device: Intel RealSense D405, Serial Number: "315122271073"
# Device: Intel RealSense D435I, Serial Number: "109622072337"

def get_rl_pipeline(serial_number):
    # camera init
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    
    # rgb = get_RGBframe(pipeline)
    # if rgb is None:
    # pipeline.stop()
    # pipeline.start(config)
    rgb = get_RGBframe(pipeline)

    return pipeline

def get_RGBframe(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color: 
        return None
    else:
        color_np = np.asanyarray(color.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        return color

def get_RGBDframe(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None
    else:
        color_np = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

        # max_distance = 2000  # Set maximum valid depth to 2000 mm
        depth_np = np.asanyarray(depth_frame.get_data())
        # print(depth_np.max(), depth_np.min())
        # depth_np[depth_np > max_distance] = 0  # Set far values to 0

        return color, depth_np


def add_pose_noise(trans, quat, trans_range, euler_range):
    # Add translation noise
    translation_noise = np.random.uniform(-trans_range, trans_range, size=3)
    noisy_translation = trans + translation_noise

    # Add rotational noise
    euler_noise = np.random.uniform(-euler_range, euler_range, size=3)  # in degrees
    rotation_noise = R.from_euler('xyz', euler_noise, degrees=True).as_quat()

    # Combine original quaternion with rotation noise
    original_rotation = R.from_quat(quat)
    noisy_rotation = original_rotation * R.from_quat(rotation_noise)

    # Convert noisy rotation back to quaternion
    noisy_quaternion = noisy_rotation.as_quat()

    return noisy_translation, noisy_quaternion


def play_trajectory(arm, rate, d_threshod, q_threshold, trans_list, quat_list, gripper_list=None, record_trajectory=False, rs_pl=None, use_noise=True):
    target_idx = 0
    done = False
    data = {'rgb':[], 'depth':[], 'translation':[], 'rotation':[], 'gripper_w':[]}

    while not done:
        done = False
        ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
        target_idx = find_next_target(ee_trans, ee_quat, trans_list, quat_list, d_threshod, q_threshold, target_idx)

        target_trans, targeta_quat = trans_list[target_idx], quat_list[target_idx]
        q_error = 1 - abs(np.dot(ee_quat, targeta_quat)) # quaternion distance
        d_error = np.linalg.norm(ee_trans - target_trans)
        if target_idx == len(trans_list) - 1 and d_error < d_threshod and q_error < q_threshold:
            done = True
        
        # if use_noise:
        #     target_trans, targeta_quat = add_pose_noise(target_trans, targeta_quat, 0.01, 2.0)

        target_rpy = R.from_quat(targeta_quat).as_euler('xyz')

        gripper_w = arm.get_gripper_width()
        # print(target_idx, len(trans_list), target_trans, ee_trans, target_rpy*180/np.pi, gripper_w)
        print(target_idx, len(trans_list), target_trans, ee_trans, gripper_w)

        # print('current rpy', left_ee_rpy*180/np.pi, target_rpy*180/np.pi)

        joint_vel = arm.robot.current_joint_velocities
        max_joint_vel = np.max(abs(joint_vel))
        if max_joint_vel > 0.3:
            arm.speed_down()
            print('slower !!!')
        else:
            arm.speed_normal()
    
        # move gripper
        if gripper_list is not None:
            arm.set_gripper_opening(gripper_list[target_idx])
        arm.set_ee_pose(target_trans, targeta_quat)

        rate.sleep()

        if record_trajectory:
            if rs_pl is not None:
                rgb = get_RGBframe(rs_pl)
                data['rgb'].append(rgb)
            ee_trans, ee_quat, ee_rpy = arm.get_ee_pose()
            gripper_w = arm.get_gripper_width()
            data['translation'].append(ee_trans)
            data['rotation'].append(ee_quat)
            data['gripper_w'].append(gripper_w)

    arm.robot.join_motion()
    return data

def get_file_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_list = os.listdir(dir_path)
    file_path = dir_path + '/' + str(len(f_list)) + '.h5'
    print('get_file_path', file_path)

    return file_path

def quaternion_distance_threshold(max_angle_deg):
    # Convert maximum angle to radians
    max_angle_rad = np.radians(max_angle_deg)
    
    # Compute the dot product threshold
    dot_product_threshold = np.cos(max_angle_rad / 2)
    
    # Compute the quaternion distance threshold
    distance_threshold = 1 - dot_product_threshold
    
    return distance_threshold