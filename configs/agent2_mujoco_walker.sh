#!/bin/bash

#SBATCH --job-name=hyper-t      #作业名称
#SBATCH --ntasks=12
#SBATCH --time=100:00:00     #申请运行时间
#SBATCH --output=./outputs/%j.out                #作业标准输出
#SBATCH --error=./outputs/%j.err                   #作业标准报错信息
#SBATCH --gres=gpu:1                   #申请1张GPU卡

# source ~/.bashrc     #激活conda环境
# conda activate dmcontrol
# python tools/retest_models.py --date_start 2024-04-03-08:43:58-532505  --date_end 2024-04-03-08:43:58-532505

# ############################################################################################################

# # ## TESAC 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# # ### CCM 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &


# # ### SaCCM 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-25-19:33:18-228153" \
#     --time_step 1000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-25-19:33:22-948946" \
#     --time_step 1000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# # # ## SaSAC 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [0], [0.75,0.85,1.0,1.15,1.25]),([0,2], [2], [0.75,0.85,1.0,1.15,1.25]), ([3,4], [2], [0.75,0.85,1.0,1.15,1.25])]" 

# ############################################################################################################

# # ### SaCCM 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-25-19:33:20-588488" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --time_step 1000000 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [0], [0.75,0.85,1.0,1.15,1.25]),([1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3], [0], [0.75,0.85,1.0,1.15,1.25]),([4], [0], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 150 \
#     --time_step 1000000 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-25-19:33:25-466950" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [0], [0.75,0.85,1.0,1.15,1.25]),([1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3], [0], [0.75,0.85,1.0,1.15,1.25]),([4], [0], [0.75,0.85,1.0,1.15,1.25])]" 

# # # ## SaSAC 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleWalkerEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [0], [0.75,0.85,1.0,1.15,1.25]),([1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3], [0], [0.75,0.85,1.0,1.15,1.25]),([4], [0], [0.75,0.85,1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name WalkerHopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 150 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0],[1.0]),([0], [2], [1.0]),([1], [2],[1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [0], [0.75,0.85,1.0,1.15,1.25]),([1], [0], [0.75,0.85,1.0,1.15,1.25]), ([3], [0], [0.75,0.85,1.0,1.15,1.25]),([4], [0], [0.75,0.85,1.0,1.15,1.25])]" 

############################################################################################################

conda run -n dmcontrol python main.py \
    --env_name WalkerHopperEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-27-06:02:20-954111" \
    --time_step 1000000 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]), ([2], [0], [1.0]), ([3], [0], [1.0]), ([4], [0], [1.0]), ([5], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0,1,2], [3], [1.0]),([1], [2], [1.0]),([0,1,2], [3], [1.0]),([0], [2], [1.0])]" & 

# SaSAC 0
conda run -n dmcontrol python main.py \
    --env_name WalkerHopperEnv \
    --env_hook DominoHook \
    --method SaSAC \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-27-06:02:23-302250" \
    --time_step 1000000 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]), ([2], [0], [1.0]), ([3], [0], [1.0]), ([4], [0], [1.0]), ([5], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0,1,2], [3], [1.0]),([1], [2], [1.0]),([0,1,2], [3], [1.0]),([0], [2], [1.0])]"  
