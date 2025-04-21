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

####################### episode length 1000 ###############################################
# # TESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-18:28:05-571513" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [0.5]),([0.5], [0.5]),([1.0], [0.5]),([2.0], [0.5]),([4.0],
#     [0.5]), ([0.1], [2.0]),([0.5], [2.0]),([1.0], [2.0]),([2.0], [2.0]),([4.0],
#     [2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1], [0.5, 0.75, 1.0]),([0.1], [0.5, 0.75, 1.0]),([4.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # CCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-18:28:06-031505" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [0.5]),([0.5], [0.5]),([1.0], [0.5]),([2.0], [0.5]),([4.0],
#     [0.5]), ([0.1], [2.0]),([0.5], [2.0]),([1.0], [2.0]),([2.0], [2.0]),([4.0],
#     [2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1], [0.5, 0.75, 1.0]),([0.1], [0.5, 0.75, 1.0]),([4.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-18:28:06-366081" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [0.5]),([0.5], [0.5]),([1.0], [0.5]),([2.0], [0.5]),([4.0],
#     [0.5]), ([0.1], [2.0]),([0.5], [2.0]),([1.0], [2.0]),([2.0], [2.0]),([4.0],
#     [2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1], [0.5, 0.75, 1.0]),([0.1], [0.5, 0.75, 1.0]),([4.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-18:28:05-791984" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [0.5]),([0.5], [0.5]),([1.0], [0.5]),([2.0], [0.5]),([4.0],
#     [0.5]), ([0.1], [2.0]),([0.5], [2.0]),([1.0], [2.0]),([2.0], [2.0]),([4.0],
#     [2.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1], [0.5, 0.75, 1.0]),([0.1], [0.5, 0.75, 1.0]),([4.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]"

# # TESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-21:10:03-910970" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([1.0], [0.5]),([1.0], [1.0]),([1.0], [4.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [4.0, 5.0]),([0.75, 1.0, 1.25], [4.0, 5.0])]" &

# # CCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-21:10:04-168939" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([1.0], [0.5]),([1.0], [1.0]),([1.0], [4.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [4.0, 5.0]),([0.75, 1.0, 1.25], [4.0, 5.0])]" &


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-21:10:03-709548" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([1.0], [0.5]),([1.0], [1.0]),([1.0], [4.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [4.0, 5.0]),([0.75, 1.0, 1.25], [4.0, 5.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-21-21:10:04-460134" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([1.0], [0.5]),([1.0], [1.0]),([1.0], [4.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [0.5, 0.75, 1.0]),([0.75, 1.0, 1.25], [4.0, 5.0]),([0.75, 1.0, 1.25], [4.0, 5.0])]"

# # ########################################### episode length 1000 ##########################################################
# # TESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-30-14:54:54-811117" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # CCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-30-14:54:08-049048" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-30-14:54:57-252030" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-30-14:54:15-206107" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]"


######################### episode length 1000 ###################################
# reward = (posafter - posbefore) / self.dt
# reward += alive_bonus
# # reward -= 1e-3 * np.square(a).sum() # control cost
# reward -= 1e-2 * np.square(a).sum() # control cost


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-31-16:24:39-943050" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-31-16:24:40-186163" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]"

# ######################### episode length 1000 ###################################
# # reward = (posafter - posbefore) / self.dt
# # reward += alive_bonus
# # # reward -= 1e-3 * np.square(a).sum() # control cost
# # reward -= 1e-1 * np.square(a).sum() # control cost


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-31-16:23:11-596528" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name HopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-07-31-16:23:11-321133" \
#     --time_step 5000000 \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0.1], [1.0]),([0.5], [1.0]),([1.0], [1.0]),([2.0], [1.0]),([4.0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0.1, 0.5, 0.75], [0.5, 0.75, 1.0]),([1.0], [0.5, 0.75, 1.0]),([1.0,1.25, 1.5, 2.0], [0.5, 0.75, 1.0]),([4.0,5.0], [0.5, 0.75, 1.0])]"



####################### CrippleHopperEnv episode length 1000 ###########################
# TESAC
conda run -n dmcontrol python main.py \
    --env_name CrippleHopperEnv \
    --env_hook DominoHook \
    --method TESAC \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-08-03-08:03:47-359533" \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [0], [1.0]),([1],[0], [1.0])]" &

# CCM
conda run -n dmcontrol python main.py \
    --env_name CrippleHopperEnv \
    --env_hook DominoHook \
    --method CCM \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-08-03-08:03:46-539511" \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [0], [1.0]),([1],[0], [1.0])]" &


# SaCCM
conda run -n dmcontrol python main.py \
    --env_name CrippleHopperEnv \
    --env_hook DominoHook \
    --method SaCCM \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-08-03-08:03:46-750028" \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [0], [1.0]),([1],[0], [1.0])]" &

# SaTESAC
conda run -n dmcontrol python main.py \
    --env_name CrippleHopperEnv \
    --env_hook DominoHook \
    --method SaSAC \
    --seed 107 \
    --save_video \
    --config_path "/home/yxue/SaMI/output/2024-08-03-08:03:47-048533" \
    --adversarial_loss_coef 0.1 \
    --buffer_size 100 \
    --contrast_batch_size 12 \
    --batch_size  12 \
    --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
    --test_eps_num_per_env 5 \
    --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [0], [1.0]),([1],[0], [1.0])]"

# ####################### CrippleHopperEnv episode length 1000 ###########################
# # TESAC
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHopperEnv \
#     --env_hook DominoHook \
#     --method TESAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-08-02-21:53:51-367440" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [2], [1.0]),([1], [2], [1.0])]" &

# # CCM
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHopperEnv \
#     --env_hook DominoHook \
#     --method CCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-08-02-21:53:50-567438" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [2], [1.0]),([1], [2], [1.0])]" &


# # SaCCM
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHopperEnv \
#     --env_hook DominoHook \
#     --method SaCCM \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-08-02-21:53:51-030531" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [2], [1.0]),([1], [2], [1.0])]" &

# # SaTESAC
# conda run -n dmcontrol python main.py \
#     --env_name CrippleHopperEnv \
#     --env_hook DominoHook \
#     --method SaSAC \
#     --seed 107 \
#     --save_video \
#     --config_path "/home/yxue/SaMI/output/2024-08-02-21:53:50-786787" \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 100 \
#     --contrast_batch_size 12 \
#     --batch_size  12 \
#     --test_envs "[([0], [0], [1.0]),([1], [0], [1.0]),([0], [2], [1.0]),([1], [2], [1.0]),([2], [0], [1.0])]" \
#     --test_eps_num_per_env 5 \
#     --train_envs "[([0], [2], [1.0]),([1], [2], [1.0]),([0], [2], [1.0]),([1], [2], [1.0])]"