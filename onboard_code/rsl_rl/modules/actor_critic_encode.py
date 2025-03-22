# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class Actor(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        scan_encoder_dims,
                        scan_decoder_dims,
                        has_scan,
                        num_scan, 
                        num_estimated,
                        actor_hidden_dims=[256, 256, 256],
                        activation='elu',
                        mu_activation= None, # If set, the last layer will be added with a special activation layer.
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Actor, self).__init__()

        self.num_lin_vel=3
        self.num_props=48
        self.num_scan=num_scan
        self.num_estimated=num_estimated
        self.num_others=self.num_estimated-self.num_lin_vel-self.num_scan

        self.has_scan = has_scan
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        # Policy
        actor_layers = []
        if self.has_scan:
            if self.if_scan_encode:
                actor_input_dim=self.num_props+scan_encoder_dims[-1]+self.num_others
                self.estimator_output_dim=self.num_lin_vel+scan_encoder_dims[-1]+self.num_others
            else:
                actor_input_dim=self.num_props+self.num_scan+self.num_others
                self.estimator_output_dim=self.num_estimated
        else:
            actor_input_dim=num_actor_obs
            self.estimator_output_dim=self.num_estimated

        # import ipdb; ipdb.set_trace()
        
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if mu_activation:
                    actor_layers.append(get_activation(mu_activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_backbone = nn.Sequential(*actor_layers)

        if self.has_scan:
            if self.if_scan_encode:
                scan_encoder = []
                scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
                scan_encoder.append(activation)
                for l in range(len(scan_encoder_dims) - 1):
                    if l == len(scan_encoder_dims) - 2:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                        scan_encoder.append(nn.Tanh())
                    else:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                        scan_encoder.append(activation)
                self.scan_encoder = nn.Sequential(*scan_encoder)
                
                self.scan_encoder_output_dim = scan_encoder_dims[-1]
                
            else:
                self.scan_encoder = nn.Identity()
                self.scan_encoder_output_dim = num_scan
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def forward(self, obs, predicted_latent=None):
        if self.has_scan:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_props:self.num_props + self.num_scan]
                if predicted_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)
                    obs_prop_scan = torch.cat([obs[:, :self.num_props], scan_latent], dim=1)
                    obs_others = obs[:, self.num_props + self.num_scan:self.num_props + self.num_scan + self.num_others]
                    backbone_input = torch.cat([obs_prop_scan, obs_others], dim=1)
                else:
                    obs_prop = obs[:, self.num_lin_vel: self.num_props]
                    obs_vel = predicted_latent[:, :self.num_lin_vel]
                    obs_scan_others = predicted_latent[:, self.num_lin_vel:]
                    backbone_input = torch.cat([obs_vel, obs_prop, obs_scan_others], dim=1)
                
            else:
                obs_scan = obs[:, self.num_props:self.num_props + self.num_scan]
                if predicted_latent is None:
                    backbone_input = obs
                else:
                    obs_prop = obs[:, self.num_lin_vel: self.num_props]
                    obs_vel = predicted_latent[:, :self.num_lin_vel]
                    obs_scan_others = predicted_latent[:, self.num_lin_vel:]
                    backbone_input = torch.cat([obs_vel, obs_prop, obs_scan_others], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if predicted_latent is None:
                backbone_input = obs
            else:
                obs_prop = obs[:, self.num_lin_vel: self.num_props]
                obs_vel = predicted_latent[:, :self.num_lin_vel]
                obs_others = predicted_latent[:, self.num_lin_vel:]
                backbone_input = torch.cat([obs_vel, obs_prop, obs_others], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def get_latent(self, obs):
        obs_vel = obs[:, :self.num_lin_vel]
        if self.has_scan:
            obs_scan = obs[:, self.num_props:self.num_props + self.num_scan]
            obs_others = obs[:, self.num_props + self.num_scan:]
            if self.if_scan_encode:
                scan_latent = self.scan_encoder(obs_scan)
            else:
                scan_latent = obs_scan
            latent = torch.cat([obs_vel, scan_latent, obs_others], dim=1)
            return latent
        else:
            obs_others = obs[:, self.num_props:]
            latent = torch.cat([obs_vel, obs_others], dim=1)
            return latent

class ActorCriticEncode(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        scan_encoder_dims,
                        scan_decoder_dims,
                        has_scan,
                        num_scan, 
                        num_estimated,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        mu_activation= None, # If set, the last layer will be added with a special activation layer.
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticEncode, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        # Policy
        self.actor=Actor(num_actor_obs, num_critic_obs, num_actions, scan_encoder_dims, scan_decoder_dims, has_scan, num_scan, num_estimated, actor_hidden_dims, activation, mu_activation)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, predicted_latent=None):
        mean = self.actor(observations, predicted_latent)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, predicted_latent=None, **kwargs):
        self.update_distribution(observations, predicted_latent)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, predicted_latent=None):
        actions_mean = self.actor(observations, predicted_latent)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def get_latent(self, obs):
        return self.actor.get_latent(obs)

    @torch.no_grad()
    def clip_std(self, min= None, max= None):
        self.std.copy_(self.std.clip(min= min, max= max))

    @property
    def estimator_output_dim(self):
        return self.actor.estimator_output_dim

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
