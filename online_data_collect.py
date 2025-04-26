import os
import yaml
import sys
import itertools
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import wandb

from logger import Manager

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

def collect_data(model, manager:Manager, hook, time_steps=-1):
    # global step counter for wandb logging
    global_step = 0

    # #############hook init#############
    hook.start_test(manager.model_parameters['train_envs'],test_envs = manager.model_parameters['test_envs'])
    # #############hook init#############
    tsne_x,tsne_y,tsne_c,tsne_alpha = [],[],[],[]
    for env_i, _env_info in tqdm(enumerate(hook.test_envs)):
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
            manager.reset_video()
            for eps_i in range(hook.max_step_num):
                print("\n","&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& eps_i ", eps_i, "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  ")
                print("previous cube pos:", observations['observation'][:,4:7])
                print("previous robot observations:", observations['observation'][:,:4])


                manager.record_video(test_env)
                # If using DummyVecEnv, extract the inner environment
                inner_env = test_env.envs[0] if hasattr(test_env, "envs") else test_env
                if inner_env.robot.gripping_succuss:
                    
                    ee_pos_bound_low_x = observations['observation'][:,4].copy() - 0.055
                    ee_pos_bound_high_x = observations['observation'][:,4].copy() + 0.06
                    pos_x = observations['observation'][0, 0].copy() # robot pos
                    ee_pos_bound_low_y = observations['observation'][:,5].copy() - 0.02
                    ee_pos_bound_high_y = observations['observation'][:,5].copy() + 0.02
                    pos_y = observations['observation'][0, 1].copy() - 0.05 # robot pos
                    pos_z = observations['observation'][0, 2].copy() # robot pos z
                    ee_pos_bound_low_z = observations['observation'][:,6].copy()
                    print("pos_x:", pos_x)
                    print("pos_y:", pos_y)
                    print("ee_pos_bound_low_x:", ee_pos_bound_low_x)
                    print("ee_pos_bound_high_x:", ee_pos_bound_high_x)
                    print("ee_pos_bound_low_y:", ee_pos_bound_low_y)
                    print("ee_pos_bound_high_y:", ee_pos_bound_high_y)
                    print("pos_z:", pos_z)
                    print("ee_pos_bound_low_z:", ee_pos_bound_low_z)
                    # Check if the robot's position is within the bounds
                    if pos_x < ee_pos_bound_low_x or pos_x > ee_pos_bound_high_x:
                        # If it's a wrong success signal
                        print("pos:", pos_x)
                        print("ee_pos_bound_low:", ee_pos_bound_low_x)
                        print("ee_pos_bound_high:", ee_pos_bound_high_x)
                        inner_env.robot.gripping_succuss = False
                    elif pos_y < ee_pos_bound_low_y or pos_y > ee_pos_bound_high_y:
                        # If it's a wrong success signal
                        print("pos_y:", pos_y)
                        print("ee_pos_bound_low_y:", ee_pos_bound_low_y)
                        print("ee_pos_bound_high_y:", ee_pos_bound_high_y)
                        inner_env.robot.gripping_succuss = False
                    elif (pos_z - ee_pos_bound_low_z) > 0.03:
                        print("pos_z:", pos_z)
                        print("ee_pos_bound_low_z:", ee_pos_bound_low_z)
                        inner_env.robot.gripping_succuss = False
                    else:
                        # If the robot is gripping the object, use a specific action
                        end_action = np.array([0.6,0.02,0.06,0.08])
                        prev_observations = copy.deepcopy(observations)
                        print("inner_env.robot.gripping_succuss:")
                        observations, rewards, dones, infos = test_env.step(end_action)
                        observations = next_observation(model,prev_observations,actions,observations, dones)
                        action = (end_action.copy() - observations['observation'][:,:4])
                        action[-1] = action[-1]/0.2
                        action[:3] = action[:3]/0.05
                        dones = np.array([True])
                else:
                    actions, states = model.predict(
                        observations,
                        state=states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    print("actions:", actions)
                    prev_observations = copy.deepcopy(observations)
                    observations, rewards, dones, infos = test_env.step(actions)
                    observations = next_observation(model,prev_observations,actions,observations, dones)


                # Log metrics to wandb if enabled
                if manager.model_parameters['use_wandb']:
                    wandb_data = {
                        'test/action': actions[0].copy(),  # Action taken
                        'test/reward': rewards[0].copy(),  
                        'test/observations': observations['observation'].copy(),  
                        'test/desired_goal': observations['desired_goal'][0].copy(),  # Desired goal
                        'test/achieved_goal': observations['achieved_goal'][0].copy(),  # Achieved goal
                        'test/steps': eps_i,  # Current step in the episode
                        'test/episode': len(hook.test_infos[hook.encoder_env_info(_env_info)]['eps_states']),  # Current episode number
                        'test/env_info': _env_info,  # Environment index
                        'test/done': dones[0].copy(),  # Done flag
                        # 'test/infos': infos[0].copy(),  # Info dictionary
                    }
                    if 'causal' in observations:
                        wandb_data['test/causal_embedding'] = observations['causal'][0]
                    # include the step index in the log
                    wandb_data['global_step'] = global_step
                    manager.wandb.log(wandb_data, step=global_step)
                    global_step += 1


                print("cube pos:", observations['observation'][:,4:7])
                print("cube rpy:", observations['observation'][:,7:10])
                print("robot observations:", observations['observation'][:,:4])
                print("goal cube:", observations['desired_goal'])
                print("rewards:", rewards)
                    

                # if ((eps_i+1) % 2 ==0 or dones) and 'hidden_h' in observations:
                if (dones) and 'hidden_h' in observations:
                    tsne_x.append(observations['causal'])
                    tsne_y.append(env_i)
                    tsne_alpha.append(1.0)
                    # tsne_alpha.append(min(eps_i/hook.max_step_num * 5, 1.0))
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

            # if _eps_states[-1] == 'success':
            manager.save_video(f'{str(_env_info)}-{len(hook.test_infos[hook.encoder_env_info(_env_info)]["eps_states"])}.mp4')
            # manager.disable_video()

            # ###########hook eps end###########
            hook.end_eps(_env_info, _eps_states)
            # ###########hook eps end###########

            # if cur_tsne == per_tsne: break

        # ###########hook env end###########
        hook.end_env(_env_info, model.logger)
        # ###########hook env end###########

        sys.stdout.flush()
        test_env.close()

    # if len(tsne_x) > 0: # 绘制tsne
    #     manager.plot_scatter(np.concatenate(tsne_x,axis=0),np.array(tsne_y),tsne_c,np.array(tsne_alpha))
    # ###########hook end###########
    hook.end_hook(manager, time_steps)
    # ###########hook end###########

    # ###########convert to csv###########
    # 在所有日志都打完之后：
    wandb.finish()  # <—— 保证上面所有 wandb.log 都同步到服务器
    api = wandb.Api()
    run = api.run(manager.wandb.path)
    print(manager.wandb.path)
    df = run.history() 
    # Append history to CSV, writing header only if file doesn't exist
    output_path = "history.csv"
    df.to_csv(
        output_path,
        mode='a',
        header=not os.path.exists(output_path),
        index=False
    )
