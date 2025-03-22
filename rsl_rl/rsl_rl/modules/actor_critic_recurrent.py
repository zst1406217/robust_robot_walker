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

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from rsl_rl.utils.collections import namedarraytuple, is_namedarraytuple

ActorCriticHiddenState = namedarraytuple('ActorCriticHiddenState', ['actor', 'critic'])
NUM_LIN_VEL=3
NUM_PROPS=49

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        scan_encoder_dims,
                        stumble_encoder_dims,
                        has_scan,
                        num_scan, 
                        num_stumble,
                        num_estimated,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        super().__init__(num_actor_obs=rnn_hidden_size,
                        num_critic_obs=rnn_hidden_size,
                        num_actions=num_actions,
                        actor_hidden_dims=actor_hidden_dims,
                        critic_hidden_dims=critic_hidden_dims,
                        activation=activation,
                        init_noise_std=init_noise_std,
                        **kwargs,
                        )
        
        self.num_lin_vel=NUM_LIN_VEL
        self.num_props=NUM_PROPS
        self.num_scan=num_scan
        self.num_stumble=num_stumble
        self.num_estimated=num_estimated
        self.num_others=self.num_estimated-self.num_lin_vel-self.num_scan

        self.has_scan = has_scan
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.if_stumble_encode = stumble_encoder_dims is not None and num_stumble > 0

        # Policy
        if self.has_scan:
            if self.if_scan_encode:
                actor_input_dim=self.num_props+scan_encoder_dims[-1]+self.num_others
                self.estimator_output_dim=self.num_lin_vel+scan_encoder_dims[-1]+self.num_others
            else:
                actor_input_dim=self.num_props+self.num_scan+self.num_others
                self.estimator_output_dim=self.num_estimated
        else:
            if self.if_stumble_encode:
                actor_input_dim=num_actor_obs-num_stumble+stumble_encoder_dims[-1]
                self.estimator_output_dim=self.num_estimated-num_stumble+stumble_encoder_dims[-1]
            else:
                actor_input_dim=num_actor_obs
                self.estimator_output_dim=self.num_estimated

        activation = get_activation(activation)

        if self.if_stumble_encode:
            stumble_encoder = []
            stumble_encoder.append(nn.Linear(num_stumble, stumble_encoder_dims[0]))
            stumble_encoder.append(activation)
            for l in range(len(stumble_encoder_dims) - 1):
                if l == len(stumble_encoder_dims) - 2:
                    stumble_encoder.append(nn.Linear(stumble_encoder_dims[l], stumble_encoder_dims[l+1]))
                    stumble_encoder.append(nn.Tanh())
                else:
                    stumble_encoder.append(nn.Linear(stumble_encoder_dims[l], stumble_encoder_dims[l + 1]))
                    stumble_encoder.append(activation)
            self.stumble_encoder = nn.Sequential(*stumble_encoder)
            print(f"Actor stumble_encoder: {self.stumble_encoder}")
            
            self.stumble_encoder_output_dim = stumble_encoder_dims[-1]
            
        else:
            self.stumble_encoder = nn.Identity()
            self.stumble_encoder_output_dim = num_stumble

        self.stumble_classifier = nn.Linear(self.stumble_encoder_output_dim, 4)

        self.memory_a = Memory(actor_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, predicted_latent=None, masks=None, hidden_states=None):

        if predicted_latent is None:
            if self.if_stumble_encode:
                obs_stumble = observations[..., -self.num_stumble:]
                obs_stumble = self.stumble_encoder(obs_stumble)
                self.stumble_class=self.stumble_classifier(obs_stumble)
                backbone_input = torch.cat([observations[..., :-self.num_stumble], obs_stumble], dim=-1)
            else:
                backbone_input = observations
        else:
            obs_prop = observations[..., self.num_lin_vel: self.num_props]
            obs_vel = predicted_latent[..., :self.num_lin_vel]
            obs_others = predicted_latent[..., self.num_lin_vel:]
            backbone_input = torch.cat([obs_vel, obs_prop, obs_others], dim=-1)

        input_a = self.memory_a(backbone_input, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations, predicted_latent=None):
        if predicted_latent is None:
            if self.if_stumble_encode:
                obs_stumble = observations[..., -self.num_stumble:]
                obs_stumble = self.stumble_encoder(obs_stumble)
                backbone_input = torch.cat([observations[..., :-self.num_stumble], obs_stumble], dim=-1)
            else:
                backbone_input = observations
        else:
            obs_prop = observations[..., self.num_lin_vel: self.num_props]
            obs_vel = predicted_latent[..., :self.num_lin_vel]
            obs_others = predicted_latent[..., self.num_lin_vel:]
            backbone_input = torch.cat([obs_vel, obs_prop, obs_others], dim=-1)
        input_a = self.memory_a(backbone_input)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return ActorCriticHiddenState(self.memory_a.hidden_states, self.memory_c.hidden_states)
    
    def get_latent(self, obs):
        obs_vel = obs[..., :self.num_lin_vel]
        if self.if_stumble_encode:
            obs_stumble = self.stumble_encoder(obs[..., -self.num_stumble:])
            obs_others = obs[..., self.num_props:-self.num_stumble]
            latent = torch.cat([obs_vel, obs_others, obs_stumble], dim=-1)
        else:
            obs_others = obs[..., self.num_props:]
            latent = torch.cat([obs_vel, obs_others], dim=-1)
        return latent
    
LstmHiddenState = namedarraytuple('LstmHiddenState', ['hidden', 'cell'])

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            if is_namedarraytuple(hidden_states):
                out, _ = self.rnn(input, tuple(hidden_states))
            else:
                out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
            return out
        else:
            # inference mode (collection): use hidden states of last step
            if is_namedarraytuple(self.hidden_states):
                out, self.hidden_states = self.rnn(input.unsqueeze(0), tuple(self.hidden_states))
            else:
                out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            if isinstance(self.hidden_states, tuple):
                self.hidden_states = LstmHiddenState(*self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0