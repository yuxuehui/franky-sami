# panda test
# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --save_video \
#     --config_path "/home/xi/yxh_space/SaMI/SaMI/output/panda_model/SaCCM"  &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --save_video \
#     --config_path "/home/xi/yxh_space/SaMI/SaMI/output/panda_model/SaTESAC"  &

# PandaPush test object_size=0.06
CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --save_video \
    --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_object_size_6cm/SaCCM_trainenv_1_1"  &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --save_video \
    --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_object_size_6cm/SaCCM_trianenv_mix"  &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaSAC \
    --save_video \
    --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_object_size_6cm/SaSAC_trainenv_1_1"  &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaSAC \
    --save_video \
    --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_object_size_6cm/SaSAC_trainenv_mix"  &

python main.py     --env_name PandaPush-v3     --env_hook PandaHook     --method SaCCM    --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_object_size_6cm/SaCCM_trainenv_1_1"


# franka robot -> PandaPush.py -> change Panda to FrankaPanda
#
# VICON cube -> PandaBase.py -> change pick_and_place to franka_pick_and_place
# i.e., change "from .tasks.pick_and_place import PickAndPlace # simulated cube
# to "from .tasks.franka_pick_and_place import PickAndPlace # VICON cube
python main.py --env_name PandaPush-v3  --env_hook PandaHook  --method SaCCM  --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_obs22_object_size_6cm/SaCCM_trainenv_1_mix"



### some note
# pinocchio
# cuRobo

# VICON
# roslaunch vicon_bridge vicon.launch
# rostopic echo 


python main.py --env_name PandaPush-v3  --env_hook PandaHook  --method SaCCM --use_wandb --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_obs22_object_size_6cm/SaCCM_trainenv_1_mix"

python main.py --env_name PandaPush-v3  --env_hook PandaHook  --method SaCCM --config_path "/home/xi/yxh_space/SaMI/SaMI/output/PandaPush_obs22_object_size_6cm/SaCCM_trainenv_mix"