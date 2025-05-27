# # panda
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 1_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 1_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(1, 1)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 1_000_000 \
#     --train_envs "[(1, 1), (1, 1), (1, 1),(1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(1, 1)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 1_000_000 \
#     --train_envs "[(1, 1), (1, 1), (1, 1),(1, 1)]" &



# # [Running 2024/4/14 19:22] panda train diferent mass
# # goal space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
# # cube space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 6_000_000 \
#     --train_envs "[(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 1),(1, 10)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 6_000_000 \
#     --train_envs "[(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 1),(1, 10)]" &


# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method TESAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(1, 1),(1, 1),(1, 1),(1, 1)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 6_000_000 \
#     --train_envs "[(1, 1),(1, 1),(1, 1),(1, 1)]" &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method TESAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(1, 1),(1, 1),(1, 1),(1, 1)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 6_000_000 \
#     --train_envs "[(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 1),(1, 10)]" &


### train with actor loss
### _train_add_actor_loss_to_encoder
# xi       2126544       1 99 15:12 pts/25   00:00:42 python ../main.py --env_name PandaPush-v3 --env_hook PandaHook --method SaCCM --adversarial_loss_coef 0.01 --buffer_size 1000 --train_freq 128 --gradient_steps 16 --learning_rate 1e-3 --batch_size 256 --contrast_batch_size 256 --encoder_tau 0.05 --seed 100 --test_envs [(0, 10), (1, 10), (10, 10), (30, 10)] --test_eps_num_per_env 50 --time_step 5_000_000 --train_envs [(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]
# xi       2126545       1 99 15:12 pts/25   00:00:42 python ../main.py --env_name PandaPush-v3 --env_hook PandaHook --method SaSAC --adversarial_loss_coef 0.01 --buffer_size 1000 --train_freq 128 --gradient_steps 16 --learning_rate 1e-3 --batch_size 256 --contrast_batch_size 256 --encoder_tau 0.05 --seed 100 --test_envs [(0, 10), (1, 10), (10, 10), (30, 10)] --test_eps_num_per_env 50 --time_step 5_000_000 --train_envs [(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &
### /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:12:29-312907
###  /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:12:29-423858

### _train_only_use_acotr_loss_to_encoder
# xi       2126792       1 99 15:13 pts/25   00:00:53 python ../main.py --env_name PandaPush-v3 --env_hook PandaHook --method SaCCM --adversarial_loss_coef 0.01 --buffer_size 1000 --train_freq 128 --gradient_steps 16 --learning_rate 1e-3 --batch_size 256 --contrast_batch_size 256 --encoder_tau 0.05 --seed 100 --test_envs [(0, 10), (1, 10), (10, 10), (30, 10)] --test_eps_num_per_env 50 --time_step 5_000_000 --train_envs [(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]
# xi       2126793       1 99 15:13 pts/25   00:00:53 python ../main.py --env_name PandaPush-v3 --env_hook PandaHook --method SaSAC --adversarial_loss_coef 0.01 --buffer_size 1000 --train_freq 128 --gradient_steps 16 --learning_rate 1e-3 --batch_size 256 --contrast_batch_size 256 --encoder_tau 0.05 --seed 100 --test_envs [(0, 10), (1, 10), (10, 10), (30, 10)] --test_eps_num_per_env 50 --time_step 5_000_000 --train_envs [(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]
### /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:13:55-836807
### /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:13:55-852701



# # train with only entropy loss
# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &
#### /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:20:08-035101
#### /home/xi/yxh_space/SaMI/SaMI/output/2025-04-29-15:20:08-410747

# train with modified target_fingers_width = np.clip(target_fingers_width, 0.0, 0.08)
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(1, 1), (1, 1), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(1, 1), (1, 1), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &



# """ [updated 2025/5/22] make sure the RL see all of the skills during training
#     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
#     2. increase adversarial_loss_coef 0.01 -> 0.1 and 1
# """
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# #     [updated 2025/5/23] make sure the RL see all of the skills during training
# #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# #     2. LSTM only input the observation, remove the action, desired_goal and achieved_goal
# #     3. use algorithms in algorithm/method_traj_distance, and Encoder
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-11:08:07-613478
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-11:08:08-225178



# #     [updated 2025/5/23] make sure the RL see all of the skills during training
# #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# #     2. LSTM only input the observation of cube, remove the action, desired_goal and achieved_goal
# #     3. use algorithms in algorithm/method_traj_distance, and EncoderCube
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-11:37:55-618887
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-11:43:03-417826



#     [updated 2025/5/23] make sure the RL see all of the skills during training
#     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
#     2. LSTM only input the height of cube (height of cube - 0.028), remove the action, desired_goal and achieved_goal
#     3. use algorithms in algorithm/method_traj_distance, and EncoderCubeHeight
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-12:00:46-289278
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-12:00:48-020047


## perfect!!
# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-14:22:14-833364
# # /home/xi/yxh_space/SaMI/SaMI/output/2025-05-23-14:22:16-543074


# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 1.0 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 3_000_000 \
#     --use_continue_train \
#     --config_path "/home/xi/yxh_space/SaMI/SaMI/saved_models/PandaPush_obs22_object_size_6cm_gripper_constraint/SaCCM_trainenv_mix_cube_height_coef_0_1_continue_training" \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 1.0 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 3_000_000 \
#     --use_continue_train \
#     --config_path "/home/xi/yxh_space/SaMI/SaMI/saved_models/PandaPush_obs22_object_size_6cm_gripper_constraint/SaSAC_trainenv_mix_cube_height_coef_0_1" \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-14:23:02-470782
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-14:23:02-500600

# CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method TESAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-27-11:00:25-314369


# #     [updated 2025/5/26] make sure the RL see all of the skills during training
# #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# #     2. Transformer only input the observation of cube, remove the action, desired_goal and achieved_goal
# #     3. use algorithms in algorithm/method_traj_distance, and EncoderActionEffect

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &


# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &


# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-15:21:56-799025
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-15:21:56-867421
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-15:21:57-225496
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-26-15:21:57-356612





# #     [updated 2025/5/27] make sure the RL see all of the skills during training
# #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# #     2. Transformer only input the height of cube, remove the action, desired_goal and achieved_goal
# #     3. use algorithms in /home/xi/yxh_space/SaMI/SaMI/algorithm/method_traj_distance_weight, and TransformerEncoderHeight
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-27-16:21:40-039164
# /home/xi/yxh_space/SaMI/SaMI/output/2025-05-27-16:21:40-269924

# # #     [updated 2025/5/27] make sure the RL see all of the skills during training
# # #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# # #     2. Attention-weighted LSTM only input the observation of cube, remove the action, desired_goal and achieved_goal
# # #     3. use algorithms in /home/xi/yxh_space/SaMI/SaMI/algorithm/method_traj_distance_transformer, and EncoderCubeAttention
# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.01 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaCCM \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
#     --env_name PandaPush-v3 \
#     --env_hook PandaHook \
#     --method SaSAC \
#     --adversarial_loss_coef 0.1 \
#     --buffer_size 1000 \
#     --train_freq 128 \
#     --gradient_steps 16 \
#     --learning_rate 1e-3 \
#     --batch_size 256 \
#     --contrast_batch_size 256 \
#     --encoder_tau 0.05 \
#     --seed 100 \
#     --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
#     --test_eps_num_per_env 50 \
#     --time_step 5_000_000 \
#     --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

# #     [updated 2025/5/27] make sure the RL see all of the skills during training
# #     1. --train_envs [(0, 30), (1, 1), (1, 30), (1, 5)]
# #     2. LSTM only input the height of cube & the distance of cube and robot, remove the action, desired_goal and achieved_goal
# #     3. use algorithms in /home/xi/yxh_space/SaMI/SaMI/algorithm/method_traj_distance_transformer, and EncoderCubeHeightDistance
CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --adversarial_loss_coef 0.01 \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaSAC \
    --adversarial_loss_coef 0.01 \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaCCM \
    --adversarial_loss_coef 0.1 \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &

CUDA_VISIBLE_DEVICES=0 nohup python ../main.py \
    --env_name PandaPush-v3 \
    --env_hook PandaHook \
    --method SaSAC \
    --adversarial_loss_coef 0.1 \
    --buffer_size 1000 \
    --train_freq 128 \
    --gradient_steps 16 \
    --learning_rate 1e-3 \
    --batch_size 256 \
    --contrast_batch_size 256 \
    --encoder_tau 0.05 \
    --seed 100 \
    --test_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 30), (1, 1), (1, 30), (1, 5)]" &