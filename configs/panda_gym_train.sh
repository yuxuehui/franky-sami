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



# [Running 2024/4/14 19:22] panda train diferent mass
# goal space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
# cube space: [0.4, -0.3, 0.03] - [0.7, 0.3, 0.03]
CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
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
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
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
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --time_step 5_000_000 \
    --train_envs "[(0, 1), (0, 5), (1, 1),(1, 5),(1, 10),(1, 2),(1, 3), (1, 1)]" &

CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
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
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --time_step 6_000_000 \
    --train_envs "[(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 1),(1, 10)]" &

CUDA_VISIBLE_DEVICES=1 nohup python ../main.py \
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
    --test_envs "[(0, 10), (1, 10), (10, 10), (30, 10)]" \
    --test_eps_num_per_env 50 \
    --time_step 6_000_000 \
    --train_envs "[(1, 1),(1, 2),(1, 3),(1, 4),(1, 5),(1, 6),(1, 1),(1, 10)]" &


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


