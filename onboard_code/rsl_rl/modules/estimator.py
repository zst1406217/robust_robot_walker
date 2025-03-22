from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm
from rsl_rl.utils import unpad_trajectories
from rsl_rl.utils.collections import namedarraytuple, is_namedarraytuple

class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        latent_encoder_hidden_dims=[256, 128],
                        activation='elu',
                        mu_activation=None,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        super(Estimator, self).__init__()

        activation = get_activation(activation)

        self.memory_a = Memory(input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(rnn_hidden_size, latent_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(latent_encoder_hidden_dims)):
            if l == len(latent_encoder_hidden_dims) - 1:
                # encoder_layers.append(nn.Linear(latent_encoder_hidden_dims[l], output_dim))
                # if mu_activation:
                #     encoder_layers.append(get_activation(mu_activation))
                pass
            else:
                encoder_layers.append(nn.Linear(latent_encoder_hidden_dims[l], latent_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
                
        self.latent_encoder = nn.Sequential(*encoder_layers)
        self.latent_output = nn.Linear(latent_encoder_hidden_dims[l], output_dim)

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a, _ = self.memory_a(observations, masks, hidden_states)
        latent=self.latent_encoder(input_a)
        predict_latent = self.latent_output(latent)
        return predict_latent

    def act_inference(self, observations):
        input_a, _ = self.memory_a(observations)
        latent=self.latent_encoder(input_a)
        predict_latent = self.latent_output(latent)
        return predict_latent
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states


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
            return out, _
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0