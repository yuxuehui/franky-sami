from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from copy import deepcopy

import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math


from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import BaseModel
from line_profiler import profile
# 
class Encoder(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        self.lstm = nn.LSTM(self.observation_dim-10, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        H,(h,c) = self.lstm(x, (h,c))
        logits = self.fc(th.relu(H)[np.arange(batch_size),0,:])
        return logits, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([obs[_x] for _x in keys]),dim = -1)
        H,(_,_) = self.lstm(x)
        logits = self.fc(th.relu(H))
        return logits

class EncoderCube(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        self.lstm = nn.LSTM(self.observation_dim - 14, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        # Slice features along last dimension before feeding to LSTM
        x_slice = x[:, :, 4:10]
        H,(h,c) = self.lstm(x_slice, (h,c))
        logits = self.fc(th.relu(H)[np.arange(batch_size),0,:])
        return logits, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat([obs[_x] for _x in keys], dim=-1)
        
        # 提取相关特征并通过Transformer
        x_slice = x[:, :, 4:10]
        H,(_,_) = self.lstm(x_slice)
        logits = self.fc(th.relu(H))
        return logits

class EncoderCubeHeight(BaseModel):
    """action_space is the dimension of the embedding"""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        self.lstm = nn.LSTM(1, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        # Slice features along last dimension before feeding to LSTM
        x_slice = x[:, :, 6:7] - 0.028
        H,(h,c) = self.lstm(x_slice, (h,c))
        logits = self.fc(th.relu(H)[np.arange(batch_size),0,:])
        return logits, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([obs[_x] for _x in keys]),dim = -1)
        # Slice features along the last dimension before feeding to LSTM
        # x shape: [batch_size, seq_len, feature_dim]
        x_slice = x[:, :, 6:7]  - 0.028
        H, (_, _) = self.lstm(x_slice)
        logits = self.fc(th.relu(H))
        return logits

class VectorQuantizer(nn.Module):
    """
    VQ-VAE style vector quantization for embedding quantization.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, embedding_dim] or [batch_size, embedding_dim]
        """
        # Flatten input for quantization if needed
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances between input and embeddings
        distances = (th.sum(flat_input**2, dim=1, keepdim=True) 
                    + th.sum(self.embedding.weight**2, dim=1)
                    - 2 * th.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest embeddings
        encoding_indices = th.argmin(distances, dim=1).unsqueeze(1)
        encodings = th.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = th.matmul(encodings, self.embedding.weight).reshape(input_shape)
        
        # Calculate VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.reshape(*input_shape[:-1])

class EncoderCubeHeightDis(BaseModel):
    """action_space is the dimension of the embedding with VQ-VAE quantization"""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3},
        num_embeddings: int = 6,  # Number of quantization embeddings
        commitment_cost: float = 0.25
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        self.lstm = nn.LSTM(1, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        
        # Add vector quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, hidden_dim, commitment_cost)
        
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        # Slice features along last dimension before feeding to LSTM
        x_slice = x[:, :, 6:7] - 0.028
        H,(h,c) = self.lstm(x_slice, (h,c))
        
        # Apply quantization during inference (no gradient computation)
        lstm_output = th.relu(H)[np.arange(batch_size),0,:]
        quantized_output, _, _ = self.vector_quantizer(lstm_output)
        # logits = self.fc(quantized_output)
        
        return quantized_output, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([obs[_x] for _x in keys]),dim = -1)
        # Slice features along the last dimension before feeding to LSTM
        # x shape: [batch_size, seq_len, feature_dim]
        x_slice = x[:, :, 6:7]  - 0.028
        H, (_, _) = self.lstm(x_slice)
        
        # Apply quantization during training
        lstm_output = th.relu(H)
        quantized_output, vq_loss, encoding_indices = self.vector_quantizer(lstm_output)
        logits = self.fc(quantized_output)
        
        # Store VQ loss for later use in training
        self.vq_loss = vq_loss
        self.encoding_indices = encoding_indices
        
        return quantized_output
    
class EncoderCubeHeightDistance(BaseModel):
    """action_space is the dimension of the embedding with VQ-VAE quantization and distance features"""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3},
        num_embeddings: int = 6,  # Number of quantization embeddings
        commitment_cost: float = 0.25
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        self.lstm = nn.LSTM(2, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        
        # Add vector quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, hidden_dim, commitment_cost)
        
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)

        # Extract cube height and normalize
        height = x[:, :, 6:7] - 0.028

        # Calculate Euclidean distance between end effector and object
        end_effector_pos = x[:, :, 0:3]  # Robot end effector position
        object_pos = x[:, :, 4:7]       # Object position 
        distance = th.norm(end_effector_pos - object_pos, dim=-1, keepdim=True)

        # Concatenate height and distance features
        x_slice = th.cat([height, distance], dim=-1)
        
        # Process through LSTM
        H,(h,c) = self.lstm(x_slice, (h,c))
        
        # Apply quantization during inference (no gradient computation)
        lstm_output = th.relu(H)[np.arange(batch_size),0,:]
        quantized_output, _, _ = self.vector_quantizer(lstm_output)
        
        return quantized_output, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([obs[_x] for _x in keys]),dim = -1)

        # Extract cube height and normalize
        height = x[:, :, 6:7] - 0.028

        # Calculate Euclidean distance between end effector and object
        end_effector_pos = x[:, :, 0:3]  # Robot end effector position
        object_pos = x[:, :, 4:7]       # Object position
        distance = th.norm(end_effector_pos - object_pos, dim=-1, keepdim=True)
        # Concatenate height and distance features
        x_slice = th.cat([height, distance], dim=-1)
        
        # Process through LSTM
        H, (_, _) = self.lstm(x_slice)
        
        # Apply quantization during training
        lstm_output = th.relu(H)
        quantized_output, vq_loss, encoding_indices = self.vector_quantizer(lstm_output)
        
        # Store VQ loss for later use in training
        self.vq_loss = vq_loss
        self.encoding_indices = encoding_indices
        
        return quantized_output

class EncoderCubeAttention(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim : int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        # Input-level attention layers
        self.attn_fc = nn.Linear((self.observation_dim - 14) + hidden_dim, self.observation_dim - 14, bias=True)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTM(self.observation_dim - 14, hidden_dim, 1,
                            bidirectional=False, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim,self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim,self.action_dim,bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat(([x[_x] for _x in keys]),dim = -1).unsqueeze(1)
        h = th.cat(([h[_h] for _h in h]),dim = -1).unsqueeze(0)
        c = th.cat(([c[_c] for _c in c]),dim = -1).unsqueeze(0)
        batch_size = x.size(0)
        # Slice features along last dimension before feeding to LSTM
        x_slice = x[:, :, 4:10]
        # Compute attention scores over input channels
        # x_slice: [batch, 1, input_dim], h: tuple of [1, batch, hidden_dim]
        prev_h = h.squeeze(0)  # [batch, hidden_dim]
        inp = x_slice.squeeze(1)  # [batch, input_dim]
        attn_input = th.cat((inp, prev_h), dim=-1)  # [batch, input_dim + hidden_dim]
        attn_scores = th.tanh(self.attn_fc(attn_input))  # [batch, input_dim]
        attn_weights = self.attn_softmax(attn_scores)  # [batch, input_dim]
        # Reweight input channels
        x_slice = (inp * attn_weights).unsqueeze(1)  # [batch, 1, input_dim]
        H,(h,c) = self.lstm(x_slice, (h,c))
        logits = self.fc(th.relu(H)[np.arange(batch_size),0,:])
        return logits, (h.squeeze(0),c.squeeze(0))

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys()) # keys = ['achieved_goal', 'desired_goal', 'observation', 'action']
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat([obs[_x] for _x in keys], dim=-1)
        
        # 提取相关特征并通过Transformer
        x_slice = x[:, :, 4:10]
        # Input-level attention before LSTM
        batch_size, seq_len, input_dim = x_slice.size()
        # Derive hidden_dim from attention layer weight
        hidden_dim = self.attn_fc.weight.shape[1] - input_dim
        # Zero previous hidden state for attention
        prev_h = x_slice.new_zeros(batch_size, hidden_dim)  # [batch, hidden_dim]
        # Flatten time steps for attention computation
        inp = x_slice.reshape(batch_size * seq_len, input_dim)  # [batch*seq_len, input_dim]
        prev_h_exp = prev_h.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, hidden_dim)
        attn_input = th.cat((inp, prev_h_exp), dim=-1)  # [batch*seq_len, input_dim+hidden_dim]
        attn_scores = th.tanh(self.attn_fc(attn_input))  # [batch*seq_len, input_dim]
        attn_weights = self.attn_softmax(attn_scores) \
            .view(batch_size, seq_len, input_dim)  # [batch, seq_len, input_dim]
        # Reweight LSTM input channels
        x_slice = x_slice * attn_weights  # [batch, seq_len, input_dim]
        H, (_, _) = self.lstm(x_slice)
        logits = self.fc(th.relu(H))
        return logits


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Transformer encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
    def forward(self, x, mask=None):
        # 清除之前的attention maps
        for hook in self.attention_hooks:
            hook.attention_maps = []

        # 如果提供了mask，确保其形状正确
        if mask is not None:
            # 确保mask的形状正确
            if mask.dim() == 2:
                # 如果是2D mask (batch_size, seq_len)，保持原样
                src_key_padding_mask = mask
            else:
                # 如果是3D mask (batch_size, seq_len, seq_len)，取最后两个维度
                src_key_padding_mask = mask[:, -1]
        else:
            src_key_padding_mask = None
            
        # Transformer编码
        output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return output
        

class TransformerEncoderHeight(BaseModel):
    """
    Encoder for action effect, using Transformer instead of LSTM.
    This encoder processes the observation of cube.
    We want to test whether the attention for cube height will be learned
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim: int = 128,
        optimizer_kwargs: dict = {'eps':1e-5, 'lr':1e-3}
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_kwargs=optimizer_kwargs
        )
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        
        self.transformer = TransformerEncoder(
            input_dim=1,  
            hidden_dim=hidden_dim,
            nhead=4,  # 多头注意力机制的头数
            num_layers=2  # Transformer层数
        )
        
        self.fc = nn.Linear(hidden_dim, self.action_dim)
        self.weight_info_nce = nn.Linear(self.action_dim, self.action_dim, bias=False)
        
    @th.no_grad()
    def forward_one_step(self, x, h, c):
        """
        Obtain the causal representation of the next step during the trajectory collection
        """
        keys = list(x.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat([x[_x] for _x in keys], dim=-1).unsqueeze(1)
        x_slice = x[:, :, 6:7]  - 0.028
        H = self.transformer(x_slice)
        logits = self.fc(th.relu(H[:, -1, :]))  # 使用最后一个时间步的输出
        
        # 创建与输入隐藏状态和单元状态相同形状的零张量
        dummy_h = th.zeros_like(h)
        dummy_c = th.zeros_like(c)
        
        return logits, (dummy_h, dummy_c)

    @profile
    def forward(self, obs):
        """
        Obtain the causal representation of entire trajectory during train
        """
        keys = list(obs.keys())
        keys.remove('achieved_goal')
        keys.remove('desired_goal')
        keys.remove('action')
        keys.sort()
        x = th.cat([obs[_x] for _x in keys], dim=-1)
        x_slice = x[:, :, 6:7]  - 0.028
        H = self.transformer(x_slice)
        logits = self.fc(th.relu(H))
        return logits
    

class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        # my params 
        causal_hidden_dim :int = 128,
        causal_out_dim: int = 6,
        exponential_smoothing:float =0.0
    ):
        self.causal_hidden_dim = causal_hidden_dim
        self.causal_out_dim = causal_out_dim
        self.action_dim = get_action_dim(action_space)
        obs_shapes = get_obs_shape(observation_space)
        self.observation_dim = sum([obs_shape[0] for obs_shape in obs_shapes.values()])
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule=lr_schedule)

        # encoder
        trajectory_space = deepcopy(self.observation_space)
        del trajectory_space.spaces['causal']
        trajectory_space = deepcopy(trajectory_space)
        trajectory_space['action'] = spaces.Box(-10,10,(self.action_dim,),dtype=np.float32)

        causal_space = spaces.Box(-10,10,(self.causal_out_dim,),dtype=np.float32)
        self.encoder = EncoderCubeAttention(trajectory_space, causal_space, hidden_dim=self.causal_hidden_dim).to(self.device)
        self.encoder_target = EncoderCubeAttention(trajectory_space, causal_space, hidden_dim=self.causal_hidden_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.encoder_target.set_training_mode(False)
        self.encoder.optimizer = self.optimizer_class(
            self.encoder.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                causal_hidden_dim = self.causal_hidden_dim,
                causal_out_dim = self.causal_out_dim,
            )
        )
        return data
    
    def rnn_encoder_predict(self, observation):
        self.set_training_mode(False)

        assert len(observation['hidden_h'].shape) == 2
        assert len(observation['observation'].shape) == 2

        causal_keys = {'hidden_c','hidden_h','causal'}
        encoder_observation = {k:v for k,v in observation.items() if k not in causal_keys}
        encoder_hidden_h = {'hidden_h': observation['hidden_h']}
        encoder_hidden_c = {'hidden_c': observation['hidden_c']}
        
        encoder_observation, _ = self.obs_to_tensor(encoder_observation)
        encoder_hidden_h, _ = self.obs_to_tensor(encoder_hidden_h)
        encoder_hidden_c, _ = self.obs_to_tensor(encoder_hidden_c)

        encoder_logits, (encoder_hidden_h, encoder_hidden_c) = \
            self.encoder.forward_one_step(encoder_observation, h=encoder_hidden_h, c=encoder_hidden_c)
        
        state = (encoder_logits.detach().cpu().numpy(),
                 encoder_hidden_h.detach().cpu().numpy(),
                 encoder_hidden_c.detach().cpu().numpy())
        return state
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        _observation = {}
        for key in observation:
            if key not in {'action', 'hidden_c','hidden_h'}:
                _observation[key] = observation[key]
        _observation, vectorized_env = self.obs_to_tensor(_observation)

        with th.no_grad():
            actions = self._predict(_observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # # Remove batch dimension if needed
        # if not vectorized_env:
        #     actions = actions.squeeze(axis=0)
        return actions, state

    def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        self.encoder.set_training_mode(mode)
        return super().set_training_mode(mode)
