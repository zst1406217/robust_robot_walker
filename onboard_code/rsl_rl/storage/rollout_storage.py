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

from collections import namedtuple
import torch

from rsl_rl.utils import split_and_pad_trajectories
from rsl_rl.utils.collections import is_namedarraytuple
from rsl_rl.utils.buffer import buffer_from_example, buffer_method, buffer_swap, buffer_expand
from rsl_rl.modules.actor_critic_recurrent import ActorCriticHiddenState

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.estimator_hidden_states = None
            self.latent = None
            self.use_gt = None
            self.transition = None
            self.env_classify = None
        
        def clear(self):
            self.__init__()

    MiniBatch = namedtuple("MiniBatch", [
        "obs",
        "critic_obs",
        "actions",
        "values",
        "advantages",
        "returns",
        "old_actions_log_prob",
        "old_mu",
        "old_sigma",
        "hidden_states",
        "estimator_hidden_states",
        "masks",
        "estimator_masks",
        "latent",
        "use_gt",
        "transition",
        "env_classify",
    ])

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, latent_shape, use_gt_shape, transition_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.latent_shape = latent_shape
        self.use_gt_shape = use_gt_shape
        self.transition_shape = transition_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.latent = torch.zeros(num_transitions_per_env, num_envs, *latent_shape, device=self.device)
        self.use_gt = torch.zeros(num_transitions_per_env, num_envs, *use_gt_shape, device=self.device)
        self.transition = torch.zeros(num_transitions_per_env, num_envs, *transition_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.env_classify = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # rnn
        self.saved_hidden_states = None
        self.saved_estimator_hidden_states = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.latent[self.step].copy_(transition.latent)
        self.use_gt[self.step].copy_(transition.use_gt)
        self.transition[self.step].copy_(transition.transition)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self._save_estimator_hidden_states(transition.estimator_hidden_states)
        self.step += 1

    def add_transitions_rnn(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.latent[self.step].copy_(transition.latent)
        self.use_gt[self.step].copy_(transition.use_gt)
        self.transition[self.step].copy_(transition.transition)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self._save_estimator_hidden_states(transition.estimator_hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """ Assuming hidden_states is a torch tensor or a namedarraytuple of torch tensor """
        if hidden_states is None:
            return
        if is_namedarraytuple(hidden_states):
            try:
                leading_dims = hidden_states.get_leading_dims()
            except AttributeError as e:
                if "NoneType" in str(e):
                    return

        if self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_from_example(hidden_states, self.observations.shape[0])
        self.saved_hidden_states[self.step] = hidden_states
        
    def _save_estimator_hidden_states(self, estimator_hidden_states):
        if estimator_hidden_states is None:
            return
        if is_namedarraytuple(estimator_hidden_states):
            try:
                leading_dims = estimator_hidden_states.get_leading_dims()
            except AttributeError as e:
                if "NoneType" in str(e):
                    return
        if self.saved_estimator_hidden_states is None:
            self.saved_estimator_hidden_states = buffer_from_example(estimator_hidden_states, self.observations.shape[0])
        self.saved_estimator_hidden_states[self.step] = estimator_hidden_states

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                # hid_batch = hid[batch_idx].unsqueeze(0)
                yield RolloutStorage.MiniBatch(
                    obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, ActorCriticHiddenState(None, None), None, None, None, None, None, None,None
                )

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_batch = buffer_method(
                    buffer_method(
                        buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[last_was_done][first_traj:last_traj],
                        "transpose", 1, 0
                    ),
                    "contiguous",
                )

                yield RolloutStorage.MiniBatch(
                    obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_batch, None, masks_batch, None, None, None, None
                )
                
                first_traj = last_traj
                
    # for estimator only
    def estimator_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop].flatten(0, 1)
                latent_batch = self.latent[:, start:stop].flatten(0, 1)
                use_gt_batch = self.use_gt[:, start:stop].flatten(0, 1)

                transition_batch = self.transition[:, start:stop].flatten(0,1)
                old_mu_batch = self.mu[:, start:stop].flatten(0, 1)
                old_sigma_batch = self.sigma[:, start:stop].flatten(0, 1)
                returns_batch = self.returns[:, start:stop].flatten(0, 1)
                advantages_batch = self.advantages[:, start:stop].flatten(0, 1)
                values_batch = self.values[:, start:stop].flatten(0, 1)
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop].flatten(0, 1)

                last_was_done = last_was_done.permute(1, 0)
                estimator_hid_batch = buffer_method(
                    buffer_method(
                        buffer_method(self.saved_estimator_hidden_states, "permute", 2, 0, 1, 3)[last_was_done][first_traj:last_traj],
                        "transpose", 1, 0
                    ),
                    "contiguous",
                )

                yield RolloutStorage.MiniBatch(
                    obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, None, estimator_hid_batch, None, masks_batch, latent_batch, use_gt_batch, transition_batch
                )
                
                first_traj = last_traj

    # for estimator + rnn actor only
    def estimator_recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories
        padded_latent, _ = split_and_pad_trajectories(self.latent, self.dones)
        padded_use_gt, _ = split_and_pad_trajectories(self.use_gt, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                latent_batch = padded_latent[:, first_traj:last_traj]
                use_gt_batch = padded_use_gt[:, first_traj:last_traj]

                transition_batch = self.transition[:, start:stop].flatten(0, 1)
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                env_classify_batch = self.env_classify[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hid_batch = buffer_method(
                    buffer_method(
                        buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[last_was_done][first_traj:last_traj],
                        "transpose", 1, 0
                    ),
                    "contiguous",
                )

                estimator_hid_batch = buffer_method(
                    buffer_method(
                        buffer_method(self.saved_estimator_hidden_states, "permute", 2, 0, 1, 3)[last_was_done][first_traj:last_traj],
                        "transpose", 1, 0
                    ),
                    "contiguous",
                )

                yield RolloutStorage.MiniBatch(
                    obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_batch, estimator_hid_batch, masks_batch, masks_batch, latent_batch, use_gt_batch, transition_batch, env_classify_batch
                )
                
                first_traj = last_traj


class QueueRolloutStorage(RolloutStorage):
    def __init__(self,
            num_envs,
            num_transitions_per_env,
            *args,
            buffer_dilation_ratio= 1.0,
            **kwargs,
        ):
        """ This rollout storage allows the buffer to be larger than the rollout length.
        NOTE: num_transitions_per_env is no longer a constant representing the buffer temporal length.
        
        Args:
            size_dilation_ratio: float, for the size of buffer bigger than num_transitions_per_env
        """
        self.num_timesteps_each_rollout = num_transitions_per_env
        self.buffer_dilation_ratio = buffer_dilation_ratio
        self.buffer_full = False
        super().__init__(
            num_envs,
            num_transitions_per_env,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def expand_buffer_once(self):
        """ Expand the buffer size in this way so that the mini_batch_generator will not output
        the buffer where no data has been stored
        """
        expand_size = int(self.buffer_dilation_ratio * self.num_timesteps_each_rollout - self.num_transitions_per_env)
        expand_size = min(expand_size, self.num_timesteps_each_rollout)
        self.num_transitions_per_env += expand_size

        # expand the buffer by concatenating
        # Core
        self.observations = torch.cat([
            self.observations,
            torch.zeros(expand_size, self.num_envs, *self.obs_shape, device=self.device),
        ], dim= 0).contiguous()
        if self.privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.cat([
                self.privileged_observations,
                torch.zeros(expand_size, self.num_envs, *self.privileged_obs_shape, device=self.device),
            ], dim= 0).contiguous()
        self.rewards = torch.cat([
            self.rewards,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.actions = torch.cat([
            self.actions,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        self.dones = torch.cat([
            self.dones,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device).byte(),
        ], dim= 0).contiguous()

        # For PPO
        self.actions_log_prob = torch.cat([
            self.actions_log_prob,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.values = torch.cat([
            self.values,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.returns = torch.cat([
            self.returns,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.advantages = torch.cat([
            self.advantages,
            torch.zeros(expand_size, self.num_envs, 1, device=self.device),
        ], dim= 0).contiguous()
        self.mu = torch.cat([
            self.mu,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        self.sigma = torch.cat([
            self.sigma,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()

        if not self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_expand(
                self.saved_hidden_states,
                expand_size,
                dim= 0,
                contiguous= True,
            )

        return expand_size

    def add_transitions(self, transition: RolloutStorage.Transition):
        return_ = super().add_transitions(transition)
        if self.step >= self.num_transitions_per_env:
            self.buffer_full = self.num_transitions_per_env >= int(self.buffer_dilation_ratio * self.num_timesteps_each_rollout)
            if self.buffer_full:
                self.step = self.step % self.num_transitions_per_env
        return return_

    def clear(self):
        """ Not return the self.step to 0 but check whether it needs to expaned the buffer.
        """
        if self.step >= self.num_transitions_per_env and not self.buffer_full:
            _ = self.expand_buffer_once() # Then self.num_transitions_per_env is updated
            print("QueueRolloutStorage: rollout storage expanded.")

    @torch.no_grad()
    def swap_from_cursor(self, buffer):
        """ This returns a new buffer (not necessarily new memory) """
        if self.step == buffer.shape[0] or self.step == 0:
            return buffer
        return torch.cat([
            buffer[self.step:],
            buffer[:self.step],
        ], dim= 0).detach().contiguous()

    def untie_buffer_loop(self):
        self.observations = self.swap_from_cursor(self.observations)
        if self.privileged_observations is not None: self.privileged_observations = self.swap_from_cursor(self.privileged_observations)
        self.actions = self.swap_from_cursor(self.actions)
        self.rewards = self.swap_from_cursor(self.rewards)
        self.dones = self.swap_from_cursor(self.dones)
        self.values = self.swap_from_cursor(self.values)
        self.actions_log_prob = self.swap_from_cursor(self.actions_log_prob)
        self.mu = self.swap_from_cursor(self.mu)
        self.sigma = self.swap_from_cursor(self.sigma)
        if not self.saved_hidden_states is None:
            with torch.no_grad():
                self.saved_hidden_states = buffer_swap(self.saved_hidden_states, self.step, contiguous= True)
        self.step = 0

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """ Re-align all the buffer to make the transitions continuous before the sampling.
        [5,6,7,8,9,0,1,2,3,4] -> [0,1,2,3,4,5,6,7,8,9] where 9 is where the latest transition stored.
        """
        if self.buffer_dilation_ratio > 1.0 and self.buffer_full:
            self.untie_buffer_loop()
        return super().reccurent_mini_batch_generator(num_mini_batches, num_epochs)

class ActionLabelRollout(QueueRolloutStorage):
    class Transition(QueueRolloutStorage.Transition):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_labels = None
    
    MiniBatch = namedtuple("MiniBatch", [
        *RolloutStorage.MiniBatch._fields,
        "action_labels",
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_labels = torch.zeros_like(self.actions)

    def expand_buffer_once(self):
        expand_size = super().expand_buffer_once()
        self.action_labels = torch.cat([
            self.action_labels,
            torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
        ], dim= 0).contiguous()
        return expand_size

    def add_transitions(self, transition: Transition):
        self.action_labels[self.step] = transition.action_labels
        return super().add_transitions(transition)

    def untie_buffer_loop(self):
        self.action_labels = self.swap_from_cursor(self.action_labels)
        return super().untie_buffer_loop()
    
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        action_labels = self.action_labels.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                action_label_batch = action_labels[batch_idx]
                yield ActionLabelRollout.MiniBatch(
                    obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None, action_label_batch,
                )

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        mini_batch_size = self.num_envs // num_mini_batches
        for idx, minibatch in enumerate(super().reccurent_mini_batch_generator(num_mini_batches, num_epochs)):
            epoch_idx = idx // num_mini_batches
            minibatch_idx = idx % num_mini_batches
            start = minibatch_idx * mini_batch_size
            stop = (minibatch_idx+1) * mini_batch_size
            action_labels_batch = self.action_labels[:, start:stop]

            yield ActionLabelRollout.MiniBatch(*minibatch, action_labels_batch)
            