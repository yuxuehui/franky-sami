from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import time

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from .off_policy_algorithm import OffPolicyAlgorithm
from .info_nce_loss import InfoNCE
from .replay_buffer import DictReplayBuffer

from line_profiler import profile
from stable_baselines3 import SAC

SelfSAC = TypeVar("SelfSAC", bound="SAC")

class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[DictReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # my_params 
        causal_keys:set = None,
        causal_hidden_dim: int = 128,
        causal_out_dim:int = 6,
        adversarial_loss_coef = 1.0,
        target_encoder_update_interval = 1,
        encoder_tau: float = 0.005,
        use_weighted_info_nce: bool = False,
        contrast_batch_size: int = 128
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            causal_keys=causal_keys,
            causal_hidden_dim=causal_hidden_dim,
            causal_out_dim=causal_out_dim,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_encoder_update_interval = target_encoder_update_interval
        self.encoder_tau = encoder_tau
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.adversarial_loss_coef = adversarial_loss_coef
        self.use_weighted_info_nce = use_weighted_info_nce
        self.contrast_batch_size = contrast_batch_size

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.encoder = self.policy.encoder
        self.encoder_target = self.policy.encoder_target

    def _process_obs(self, observations,add_causal=True,add_hidden=False,add_action=False):
        _observations = {}
        key_order = ['action','observation','causal','hidden_c','hidden_h']
        key_special = {'action','causal','hidden_c','hidden_h'}
        for key in key_order:
            if add_hidden and (key == 'hidden_h' or key == 'hidden_c'):
                _observations[key] = observations[key]
            if add_causal and (key == 'causal'):
                _observations[key] = observations[key]
            if add_action and (key == 'action'):
                _observations[key] = observations[key]
            if key not in self.causal_keys | key_special:
                _observations[key] = observations[key]
        return _observations
    
    @profile
    def _train(self,gradient_step,batch_size):
        if self.contrast_batch_size > 0:
            replay_data = self.replay_buffer.sample_contrast(self.contrast_batch_size)  # type: ignore[union-attr]
            loss_fn = InfoNCE()
            pt, pr, nt, nr = replay_data.pos_trajectories, replay_data.pos_trajectory_rewards, replay_data.neg_trajectories, replay_data.neg_trajectory_rewards
            encoder_loss = []
            weights = []
            for positive_traj, positive_reward, negative_traj, negative_reward in zip(pt, pr, nt, nr):
                weight = positive_reward.mean() - negative_reward.mean()

                query = self.encoder(positive_traj)
                with th.no_grad():
                    positive_key = self.encoder_target(positive_traj)
                    negative_key = self.encoder_target(negative_traj)

                if self.use_weighted_info_nce:
                    positive_key = self.encoder.weight_info_nce(positive_key)
                    negative_key = self.encoder.weight_info_nce(negative_key)
                
                encoder_loss.append(weight * loss_fn(query, positive_key, negative_key))
                weights.append(weight)
            encoder_loss = th.stack(encoder_loss).mean()
            weights = th.stack(weights).mean()
            self.encoder_losses.append(encoder_loss.item())
            self.weights.append(weights.item())
        else:
            encoder_loss = 0
            self.encoder_losses.append(0)
            self.weights.append(0)
        
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        # Log detailed metrics
        if self.logger is not None:
            with th.no_grad():
                # Log critic predictions
                observations['causal'] = causal.detach()
                current_q_values = self.critic(observations, replay_data.actions)
                mean_q = th.cat(current_q_values, dim=1).mean().item()
                self.logger.record("train/q_value", mean_q)
                
                # Log state-action pairs and their values
                actions_pi, log_prob = self.actor.action_log_prob(observations)
                mean_action = actions_pi.mean().item()
                mean_log_prob = log_prob.mean().item()
                self.logger.record("train/actions", mean_action)
                self.logger.record("train/log_prob", mean_log_prob)
                
                # Log target network predictions
                next_q_values = th.cat(self.critic_target(next_observations, next_actions), dim=1)
                mean_target_q = next_q_values.mean().item()
                self.logger.record("train/target_q_value", mean_target_q)

        # We need to sample because `log_std` may have changed between two gradient steps
        if self.use_sde:
            self.actor.reset_noise()

        observations = self._process_obs(replay_data.observations,add_causal=False,add_hidden=True,add_action=True)
        next_observations = self._process_obs(replay_data.next_observations,add_causal=False,add_hidden=True,add_action=True)
        
        causal = self.encoder(observations)
        next_causal = self.encoder_target(next_observations)

        observations = self._process_obs(replay_data.observations)
        # Action by the current actor for the sampled state
        observations['causal'] = causal.detach()
        actions_pi, log_prob = self.actor.action_log_prob(observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        self.ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_observations = self._process_obs(replay_data.next_observations)
            next_observations['causal'] = next_causal
            next_actions, next_log_prob = self.actor.action_log_prob(next_observations)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.critic_target(next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        observations['causal'] = causal
        current_q_values = self.critic(observations, replay_data.actions)

        # Compute critic loss 
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        self.critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.encoder.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        (self.adversarial_loss_coef * encoder_loss + critic_loss).backward()
        self.critic.optimizer.step()
        self.encoder.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        for key in observations:
            observations[key] = observations[key].detach()
        q_values_pi = th.cat(self.critic(observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        self.actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if gradient_step % self.target_update_interval == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        if gradient_step % self.target_encoder_update_interval == 0:
            polyak_update(self.encoder.parameters(), self.encoder_target.parameters(), self.encoder_tau)
    
    def reset_losses(self,):
        self.ent_coef_losses = []
        self.ent_coefs = []
        self.actor_losses = []
        self.critic_losses = []
        self.encoder_losses = [] 
        self.weights = []

    @profile
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer,
                      self.encoder.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        self.reset_losses()
        
        for gradient_step in range(gradient_steps):
            self._train(gradient_step,batch_size)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(self.ent_coefs))
        self.logger.record("train/actor_loss", np.mean(self.actor_losses))
        self.logger.record("train/critic_loss", np.mean(self.critic_losses))
        self.logger.record("train/weight", np.mean(self.weights))
        if len(self.ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(self.ent_coef_losses))
        self.logger.record("train/encoder_loss", np.mean(self.encoder_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target","encoder"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer","encoder.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables