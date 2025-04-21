#!/bin/bash

#SBATCH --job-name=hyper-t      #作业名称
#SBATCH --ntasks=6
#SBATCH --time=100:00:00     #申请运行时间
#SBATCH --output=./outputs/%j.out                #作业标准输出
#SBATCH --error=./outputs/%j.err                   #作业标准报错信息
#SBATCH --gres=gpu:1                   #申请1张GPU卡

# source ~/.bashrc     #激活conda环境
# conda activate dmcontrol
# python tools/retest_models.py --date_start 2024-04-03-08:43:58-532505  --date_end 2024-04-03-08:43:58-532505

#################### resize  / change reward ###########################
# reward_ctrl = -0.01 * np.sum(np.square(act), axis=-1)


# ### SaCCM 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  128 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([0],[2],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([1],[2],[0.75,0.85,1.0,1.15,1.25]), ([2],[2],[0.75,0.85,1.0,1.15,1.25]), ([3],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25]), ([1],[0],[0.75,0.85,1.0,1.15,1.25]), ([2],[0],[0.75,0.85,1.0,1.15,1.25]), ([3],[0],[0.75,0.85,1.0,1.15,1.25]), ([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([1],[2],[0.75,0.85,1.0,1.15,1.25]), ([2],[2],[0.75,0.85,1.0,1.15,1.25]), ([3],[2],[0.75,0.85,1.0,1.15,1.25])]"  


# ## SaSAC 0
# conda run -n dmcontrol python ./main.py \
#     --env_name CrippleAntEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([0],[2],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([1],[2],[0.75,0.85,1.0,1.15,1.25]), ([2],[2],[0.75,0.85,1.0,1.15,1.25]), ([3],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25]), ([1],[0],[0.75,0.85,1.0,1.15,1.25]), ([2],[0],[0.75,0.85,1.0,1.15,1.25]), ([3],[0],[0.75,0.85,1.0,1.15,1.25]), ([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([1],[2],[0.75,0.85,1.0,1.15,1.25]), ([2],[2],[0.75,0.85,1.0,1.15,1.25]), ([3],[2],[0.75,0.85,1.0,1.15,1.25])]" 

### SaCCM 0
conda run -n dmcontrol python ./main.py \
    --env_name CrippleAntEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-30-19:51:15-137401" \
    --time_step 5000000 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([0],[2],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25]), ([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25])]"  & 


## SaSAC 0
conda run -n dmcontrol python ./main.py \
    --env_name CrippleAntEnv \
    --env_hook DominoHook \
    --method SaSAC \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-30-19:51:32-217520" \
    --time_step 5000000 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 1000 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0],[1.0]),([1], [0],[1.0]),([2], [0],[1.0]),([0],[2],[1.0]),([0, 3], [1], [1.0]),([1, 3], [1], [1.0]),([1, 2], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25]), ([0],[2],[0.75,0.85,1.0,1.15,1.25]), ([0],[0],[0.75,0.85,1.0,1.15,1.25])]" 

