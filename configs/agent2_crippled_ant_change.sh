#!/bin/bash

#SBATCH --job-name=hyper-t      #作业名称
#SBATCH --ntasks=28
#SBATCH --time=100:00:00     #申请运行时间
#SBATCH --output=./outputs/%j.out                #作业标准输出
#SBATCH --error=./outputs/%j.err                   #作业标准报错信息
#SBATCH --gres=gpu:1                   #申请1张GPU卡

# source ~/.bashrc     #激活conda环境
# conda activate dmcontrol
# python tools/retest_models.py --date_start 2024-04-03-08:43:58-532505  --date_end 2024-04-03-08:43:58-532505

################################################################################################

# ## TESAC 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --save_video \
#     --use_continue_train \
#     --config_path "/home/yxue/SaMI/output/2024-07-22-08:03:52-135354" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([3], [0],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85,1.0,1.15,1.25]), ([2, 3], [0], [0.75,0.85,1.0,1.15,1.25]), ([0, 3], [1], [1.0,1.15,1.25]), ([0, 3], [1], [0.75,0.85])]" &


# ### CCM 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --save_video \
#     --use_continue_train \
#     --config_path "/home/yxue/SaMI/output/2024-07-22-08:03:51-913136" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([3], [0],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85,1.0,1.15,1.25]), ([2, 3], [0], [0.75,0.85,1.0,1.15,1.25]), ([0, 3], [1], [1.0,1.15,1.25]), ([0, 3], [1], [0.75,0.85])]" &


# ### SaCCM 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --use_continue_train \
#     --config_path "/home/yxue/SaMI/output/2024-07-22-08:03:52-650841" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([3], [0],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85,1.0,1.15,1.25]), ([2, 3], [0], [0.75,0.85,1.0,1.15,1.25]), ([0, 3], [1], [1.0,1.15,1.25]), ([0, 3], [1], [0.75,0.85])]"  &


# ### SaSAC 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --use_continue_train \
#     --config_path "/home/yxue/SaMI/output/2024-07-22-08:03:52-369707" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([3], [0],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0, 1], [0], [0.75,0.85,1.0,1.15,1.25]), ([2, 3], [0], [0.75,0.85,1.0,1.15,1.25]), ([0, 3], [1], [1.0,1.15,1.25]), ([0, 3], [1], [0.75,0.85])]" 

############################################################################################################

