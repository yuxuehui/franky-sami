import os
import yaml
import sys
import itertools
import copy
import numpy as np
import torch as th
from tqdm import tqdm
from collections import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv
import time

from logger import Manager
from markovianess.ci.conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

def next_observation(model, prev_observations,actions,observations, dones):
    """
    Return: observation
        observations['causal']: context embedding
        observations['hidden_h']: hidden state
        observations['hidden_c']: cell state

    prev_observations: Previous observations
    actions: Actions
    observations: Observations
    dones: Dones
    """
    if 'hidden_h' in observations:
        # for rnn 
        # reset next obs hidden_h and hidden_c
        _observations = OrderedDict()
        actions = model.policy.scale_action(actions)
        _observations['action'] = (actions * (1-np.stack((dones,), axis = -1))).astype(np.float32)
        for key in observations:
            _observations[key] = observations[key].astype(np.float32)
        _observations['hidden_h'] = prev_observations['hidden_h']
        _observations['hidden_c'] = prev_observations['hidden_c']
        causal, hidden_h, hidden_c = model.policy.rnn_encoder_predict(_observations)
        observations['causal'] = causal.astype(np.float32)
        observations['hidden_h'] = hidden_h * (1-np.stack((dones,), axis = -1)).astype(np.float32)
        observations['hidden_c'] = hidden_c * (1-np.stack((dones,), axis = -1)).astype(np.float32)
    
    return observations

def test_model(model, manager:Manager, hook, time_steps=-1):
    # #############hook init#############
    hook.start_test(manager.model_parameters['train_envs'],test_envs = manager.model_parameters['test_envs'])
    # #############hook init#############
    tsne_x,tsne_y,tsne_c,tsne_alpha = [],[],[],[]
    markov_scores = {}  # 保存每个环境的分数
    for env_i, _env_info in tqdm(enumerate(hook.test_envs)):
        all_observations = []  # 每个环境单独收集
        # test env
        env = hook.make_env(manager, _env_info)
        test_env = DummyVecEnv([env])
        test_env.envs[0].env.render()
        # ###########hook env start###########
        hook.start_env(_env_info)
        # ###########hook env start###########
        if manager.model_parameters['save_video']:
            manager.enable_video()
        else:
            manager.disable_video()
        while len(hook.test_infos[hook.encoder_env_info(_env_info)]['eps_states']) < manager.model_parameters['test_eps_num_per_env']:
            observations = test_env.reset()
            print("reset successful!!!")
            states = None
            episode_starts = np.ones((test_env.num_envs,), dtype=bool)
            _eps_states = []
            episode_observations = []  # 收集单个episode的observations
            manager.reset_video()
            for eps_i in range(hook.max_step_num):
                manager.record_video(test_env)
                actions, states = model.predict(
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                prev_observations = copy.deepcopy(observations)
                observations, rewards, dones, infos = test_env.step(actions)
                observations = next_observation(model,prev_observations,actions,observations, dones)
                # 收集每步的obs
                if isinstance(observations, dict) and 'causal' in observations:
                    temp_obs = observations.copy()
                    keys = list(temp_obs.keys())
                    keys.remove('achieved_goal')
                    keys.remove('desired_goal')
                    keys.remove('hidden_h')
                    keys.remove('hidden_c')
                    x = th.cat([th.from_numpy(temp_obs[_x]) if isinstance(temp_obs[_x], np.ndarray) else temp_obs[_x] for _x in keys], dim=-1)
                    episode_observations.append(x[0][:-6].clone())
                if ((eps_i+1) % 2 ==0 or dones) and 'hidden_h' in observations:
                    tsne_x.append(observations['causal'])
                    tsne_y.append(env_i)
                    tsne_alpha.append(min(eps_i/hook.max_step_num * 5, 1.0))
                    class_name = hook.encoder_env_info(_env_info)
                    if class_name not in tsne_c:
                        tsne_c.append(class_name)
                if not dones:
                    _eps_states.append(hook.get_state(test_env, infos))
                else:
                    if infos[0]['is_success']:
                        _eps_states.append('success')
                    else:
                        _eps_states.append('fail')
                    break
            if len(episode_observations) > 0:
                all_observations.extend(episode_observations)
            manager.save_video(f'{str(_env_info)}-{len(hook.test_infos[hook.encoder_env_info(_env_info)]["eps_states"])}.mp4')
            hook.end_eps(_env_info, _eps_states)
        hook.end_env(_env_info, model.logger)
        sys.stdout.flush()
        test_env.close()
        # 每个环境结束后计算Markov violation score
        if len(all_observations) > 0:
            obs_array = np.stack([x.detach().cpu().numpy() for x in all_observations], axis=0)
            print(f"[{_env_info}] obs_array shape: {obs_array.shape}")
            # Robust checks before PCMCI
            min_timesteps = 10  # or 2 * tau_max
            if obs_array.ndim != 2:
                print(f"[{_env_info}] Skipping PCMCI: obs_array should be 2D, got {obs_array.shape}")
                continue
            if obs_array.shape[0] < min_timesteps:
                print(f"[{_env_info}] Skipping PCMCI: not enough time steps ({obs_array.shape[0]} < {min_timesteps})")
                continue
            if obs_array.shape[1] < 2:
                print(f"[{_env_info}] Skipping PCMCI: not enough variables ({obs_array.shape[1]} < 2)")
                continue
            if np.isnan(obs_array).any() or np.isinf(obs_array).any():
                print(f"[{_env_info}] Skipping PCMCI: obs_array contains NaN or Inf")
                continue
            try:
                cit = ConditionalIndependenceTest()
                results_dict = cit.run_pcmci(
                    observations=obs_array,
                    tau_max=4,
                    alpha_level=0.05
                )
                p_matrix = results_dict["p_matrix"]
                val_matrix = results_dict["val_matrix"]
                markov_score = get_markov_violation_score(
                    p_matrix=p_matrix,
                    val_matrix=val_matrix,
                    alpha_level=0.05
                )
                print(f"[{_env_info}] Markov violation score: {markov_score:.6f}")
                markov_scores[str(_env_info)] = markov_score
            except Exception as e:
                print(f"[{_env_info}] PCMCI computation failed: {e}")
        else:
            print(f"[{_env_info}] No observations collected, skipping PCMCI.")
    print("All Markov violation scores by env:")
    for env, score in markov_scores.items():
        print(f"{env}: {score:.6f}")
    if len(tsne_x) > 0:
        manager.plot_scatter(np.concatenate(tsne_x,axis=0),np.array(tsne_y),tsne_c,np.array(tsne_alpha))
    hook.end_hook(manager, time_steps)
