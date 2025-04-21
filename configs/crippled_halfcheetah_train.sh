#!/bin/bash

#SBATCH --job-name=hyper-t      #作业名称
#SBATCH --ntasks=20
#SBATCH --time=100:00:00     #申请运行时间
#SBATCH --output=./outputs/%j.out                #作业标准输出
#SBATCH --error=./outputs/%j.err                   #作业标准报错信息
#SBATCH --gres=gpu:1                   #申请1张GPU卡

# source ~/.bashrc     #激活conda环境
# conda activate dmcontrol
# python tools/retest_models.py --date_start 2024-04-03-08:43:58-532505  --date_end 2024-04-03-08:43:58-532505

# ############################################################################################################

# # TESAC 0
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:16-390935" \
#     --test_envs "[([0], [0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:16-618838" \
#     --test_envs "[([1], [0], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([1], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:17-485915" \
#     --test_envs "[([2], [0], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([2], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:17-053418" \
#     --test_envs "[([3], [0], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([3], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:17-271214" \
#     --test_envs "[([4], [0], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([4], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-10:38:16-866709" \
#     --test_envs "[([5], [0], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([5], [0], [1.0,1.15,1.25])]" 
# ############################################################################################################

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-13:22:26-818000" \
#     --test_envs "[([4,5], [1], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([4,5], [1], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-13:22:26-604499" \
#     --test_envs "[([1,2], [1], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([1,2], [1], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-13:22:26-368617" \
#     --test_envs "[([0,3], [1], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0,3], [1], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-13:22:26-150386" \
#     --test_envs "[([1,4], [1], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([1,4], [1], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-14-13:22:27-030092" \
#     --test_envs "[([2,5], [1], [1.0,1.15,1.25])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([2,5], [1], [1.0,1.15,1.25])]" &
# # ############################################################################################################

# # # TESAC
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  128 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-13-14:11:59-329187" \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" & 

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-13-14:11:59-527784" \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" & 

# # CCM
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  128 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-13-14:11:59-814409" \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" &

# conda run -n dmcontrol python main.py \
#     --env_name CrippleHalfCheetahEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --use_wandb \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-13-14:12:00-122088" \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" 

# SaCCM
conda run -n dmcontrol python main.py \
    --env_name CrippleHalfCheetahEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --seed 107 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  128 \
    --use_wandb \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-14-00:57:54-895625" \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]"  &

conda run -n dmcontrol python main.py \
    --env_name CrippleHalfCheetahEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --seed 107 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --use_wandb \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-14-00:57:54-201569" \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]"  &

# SaTESAC
conda run -n dmcontrol python main.py \
    --env_name CrippleHalfCheetahEnv \
    --env_hook DominoHook \
    --method SaSAC \
    --seed 107 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  128 \
    --use_wandb \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-14-00:57:54-438134" \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" &

conda run -n dmcontrol python main.py \
    --env_name CrippleHalfCheetahEnv \
    --env_hook DominoHook \
    --method SaSAC \
    --seed 107 \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --use_wandb \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-07-14-00:57:54-692382" \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([2], [0], [1.0]),([3], [0], [1.0]),([4], [0], [1.0]), ([4,5], [1], [1.0]), ([4,5], [1], [1.0]),([1,4], [1], [1.0]),([0,3], [1], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([3], [0], [1.0,1.15,1.25]), ([3], [0], [1.0,1.15,1.25]), ([4], [0], [1.0,1.15,1.25]), ([5], [0], [1.0,1.15,1.25])]" 