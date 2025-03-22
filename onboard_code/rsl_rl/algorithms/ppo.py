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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_paras,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 clip_min_std= 1e-15, # clip the policy.std if it supports, check update()
                 optimizer_class_name= "Adam",
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device
        self.ppo_train_together = estimator_paras["ppo_train_together"]
        self.estimator_loss_coeff = estimator_paras["estimator_loss_coeff"]

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_min_std = torch.tensor(clip_min_std, device= self.device) if isinstance(clip_min_std, (tuple, list)) else clip_min_std
        
        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.estimator_learning_rate=estimator_paras["learning_rate"]
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        # self.scan_vae_loss=estimator_paras["scan_vae_loss"]
        self.train_together = estimator_paras["train_together"]
        self.gt_ratio = 1.
        
        if self.ppo_train_together:
            self.optimizer = getattr(optim, optimizer_class_name)(list(self.actor_critic.parameters())+list(self.estimator.parameters()), lr=learning_rate)
        else:
            self.optimizer = getattr(optim, optimizer_class_name)(self.actor_critic.parameters(), lr=learning_rate)
            self.estimator_optimizer = optim.AdamW(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        
        print(f"Estimator: {self.estimator}")
        
        # algorithm status
        self.current_learning_iteration = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
        self.estimator.test()
    
    def train_mode(self):
        self.actor_critic.train()
        self.estimator.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        self.transition.estimator_hidden_states = self.estimator.get_hidden_states()
        obs_real=obs.clone()
        predicted_latent=self.estimator.act(obs_real[:, 3:48]).squeeze(0)
        latent_gt=self.actor_critic.get_latent(obs_real)
        
        if self.train_together:
            env_num=obs.shape[0]
            use_gt=torch.zeros([env_num, 1]).to(self.device)
            use_gt[torch.where(torch.rand([env_num, 1]).to(self.device)<self.gt_ratio)] = 1
            latent=latent_gt * use_gt + predicted_latent * (1-use_gt)
            self.transition.actions = self.actor_critic.act(obs, latent).detach()
            self.transition.latent = latent
            self.transition.use_gt = use_gt
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()
            use_gt=torch.ones([env_num, 1]).to(self.device)
            self.transition.latent = latent_gt
            self.transition.use_gt = use_gt
            
        # Compute the actions and values

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def act_play(self, obs):
        obs_real=obs.clone()
        predicted_latent=self.estimator.act_inference(obs_real[:, 3:48]).squeeze(0)
        # latent_gt=self.actor_critic.get_latent(obs_real)
        
        if self.train_together:
            actions = self.actor_critic.act_inference(obs, predicted_latent).detach()
        else:
            actions = self.actor_critic.act_inference(obs).detach()
        # Compute the actions and values
        return actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        self.estimator.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(lambda :0.)
        average_stats = defaultdict(lambda :0.)
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            # generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            generator = self.storage.estimator_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for minibatch in generator:
            losses, _, stats = self.compute_losses(minibatch)
            loss = 0.
            for k, v in losses.items():
                loss += getattr(self, k + "_coef", 1.) * v
                mean_losses[k] = mean_losses[k] + v.detach()
            mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
            for k, v in stats.items():
                average_stats[k] = average_stats[k] + v.detach()

            # Gradient step
            self.optimizer.zero_grad()
            if not self.ppo_train_together:
                self.estimator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if not self.ppo_train_together:
                self.estimator_optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            average_stats[k] = average_stats[k] / num_updates
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min= self.clip_min_std)

        return mean_losses, average_stats

    def compute_losses(self, minibatch):
        obs_real=minibatch.obs.clone()
        obs_unpad=unpad_trajectories(minibatch.obs, minibatch.estimator_masks).flatten(0,1)
        scan_obs=obs_unpad[:,48:48+187]
        critic_obs_unpad=unpad_trajectories(minibatch.critic_obs, minibatch.estimator_masks).flatten(0,1)
        critic_scan_obs=critic_obs_unpad[:,48:48+187]
        predicted_latent=self.estimator.act(obs_real[:, :, 3:48], masks=minibatch.estimator_masks, hidden_states=minibatch.estimator_hidden_states).flatten(0,1)
        with torch.inference_mode():
            latent_gt=self.actor_critic.get_latent(critic_obs_unpad)
            latent_obs=self.actor_critic.get_latent(obs_unpad)
        latent_used=latent_obs * minibatch.use_gt + predicted_latent * (1-minibatch.use_gt)
        # predicted_scan, mu, logvar=self.actor_critic.actor.reconstruct_scan(scan_obs)
        
        if self.train_together:
            if self.ppo_train_together:
                self.actor_critic.act(obs_unpad, latent_used).detach()
            else:
                self.actor_critic.act(obs_unpad, minibatch.latent).detach()
            scan_loss=torch.tensor(0.)
        else:
            self.actor_critic.act(obs_unpad).detach()
            # scan_loss_mse=(predicted_scan - critic_scan_obs).pow(2).mean()
            # scan_loss_kld=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp()) / mu.shape[0]
            # scan_loss=0.01*(scan_loss_mse+self.scan_vae_loss*scan_loss_kld)
            scan_loss=torch.tensor(0.)
            
        if self.train_with_estimated_states:
            # Estimator
            estimator_loss = (predicted_latent - latent_gt).pow(2).mean()*self.estimator_loss_coeff
        else:
            estimator_loss = torch.tensor(0.)
        
        # self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hidden_states.actor)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        value_batch = self.actor_critic.evaluate(critic_obs_unpad, masks=minibatch.masks, 
                                                 hidden_states=minibatch.hidden_states.critic if minibatch.hidden_states is not None else None)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                            torch.log(sigma_batch / minibatch.old_sigma + 1.e-5) + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                    
        # for param_group in self.estimator_optimizer.param_groups:
        #     param_group['lr']=self.estimator_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        surrogate = -torch.squeeze(minibatch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(minibatch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param,
                                                                                                    self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (minibatch.returns - value_batch).pow(2).mean()
        
        return_ = dict(
            surrogate_loss= surrogate_loss,
            value_loss= value_loss,
            estimator_loss=estimator_loss,
            scan_loss=scan_loss,
        )
        if entropy_batch is not None:
            return_["entropy"] = - entropy_batch.mean()
        
        inter_vars = dict(
            ratio= ratio,
            surrogate= surrogate,
            surrogate_clipped= surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == 'adaptive':
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped
        return return_, inter_vars, dict()
