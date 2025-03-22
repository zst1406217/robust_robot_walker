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

from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from collections import OrderedDict, defaultdict
from copy import copy

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.math import quat_apply_yaw
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = getattr(self.cfg.viewer, "debug_viz", False)
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions):
        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_low", None), (tuple, list)):
            self.cfg.normalization.clip_actions_low = torch.tensor(
                self.cfg.normalization.clip_actions_low,
                device= self.device
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_high", None), (tuple, list)):
            self.cfg.normalization.clip_actions_high = torch.tensor(
                self.cfg.normalization.clip_actions_high,
                device= self.device
            )
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        # some customized action clip methods to bound the action output
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
        elif getattr(self.cfg.normalization, "clip_actions_method", None) == "hard":
            actions_low = getattr(
                self.cfg.normalization, "clip_actions_low",
                self.dof_pos_limits[:, 0] - self.default_dof_pos,
            )
            actions_high = getattr(
                self.cfg.normalization, "clip_actions_high",
                self.dof_pos_limits[:, 1] - self.default_dof_pos,
            )
            self.actions = torch.clip(actions, actions_low, actions_high)
        else:
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    def post_decimation_step(self, dec_i):
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        
        px=self.root_states[:, 0]
        py=self.root_states[:, 1]
        px=torch.round(px/self.terrain.cfg.horizontal_scale).long()
        py=torch.round(py/self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        self.base_height[:] = (self.root_states[:, 2] - self.height_samples[px, py]* self.terrain.cfg.vertical_scale).unsqueeze(1)
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        if self.cfg.env.goal_command:
            self.time_info[:, 1]=self.time_info[:, 1]-self.dt
            root_states = self.root_states[:, :2]-self.env_origins[:, :2]-self.base_init_state[:2]
            _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
            now_states = torch.cat([root_states, yaw.unsqueeze(-1)], dim=1) # [num_env, 3]
            self.delta_goal = self.goal_commands-now_states
            self.delta_yaw = self.delta_goal[:, 2]
            self.delta_yaw = self.delta_yaw % (2*np.pi)
            self.delta_yaw[self.delta_yaw>np.pi] -= np.pi * 2
            self.delta_goal[:, 2]=self.delta_yaw
            self.delta_xy = self.delta_goal[:, :2].clone()
            self.delta_goal[:, 0]=self.delta_xy[:,0]*torch.cos(yaw)+self.delta_xy[:,1]*torch.sin(yaw)
            self.delta_goal[:, 1]=-self.delta_xy[:,0]*torch.sin(yaw)+self.delta_xy[:,1]*torch.cos(yaw)
            self.delta_goal[:, 2] = 0
            goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
            self.delta_goal[:, 0]*=(goal_dis>0.2)*(self.time_info[:, 1]<self.start_time)
            self.delta_goal[:, 1]*=(goal_dis>0.2)*(self.time_info[:, 1]<self.start_time)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_transition_states()
            
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_torques[:] = self.torques[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        self.last_base_height[:] = self.base_height[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf=self.time_out_buf.clone()

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._init_goal_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)
        if self.cfg.env.goal_command:
            self._init_goal_commands(env_ids)
            
        self._reset_buffers(env_ids)
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _get_proprio_with_goal_obs(self, privileged = False):
        # always 49 dim
        obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.delta_goal[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.time_info[:, 1:]
                                    ),dim=-1)
        if (not privileged) and (not getattr(self.cfg.env, "use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        if privileged and (not getattr(self.cfg.env, "privileged_use_lin_vel", True)):
            obs_buf[:, :3] = 0.
        obs_buf[:, 48]*=(self.time_info[:, 1]<self.start_time)
        return obs_buf

    def _get_height_measurements_obs(self, privileged = False):
        # not tested, should be fixed at 187 dim, otherwise, check your config
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) + self.height_measurement_offset - self.measured_heights, -1, 1.)
        return heights

    def _get_base_pose_obs(self, privileged = False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        return torch.cat([
            self.root_states[:, :3] - self.env_origins,
            torch.stack([roll, pitch, yaw], dim= -1),
        ], dim= -1)
    
    def _get_robot_config_obs(self, privileged = False):
        return self.robot_config_buffer
    
    def _get_robot_friction_obs(self, privileged = False):
        return self.robot_config_buffer[:, 0].unsqueeze(1)-1.
    
    def _get_stumble_state_obs(self, privileged = False):
        if self.cfg.asset.step_num>0:
            num_div = 100/self.cfg.asset.step_num
            body_contact = (torch.norm(self.contact_forces[:, self.collision_body_indices, :], dim=-1)//num_div*num_div/100).float()
            body_contact = torch.clip(body_contact*2.-1., -1., 1.)
        else:
            body_contact = (torch.norm(self.contact_forces[:, self.collision_body_indices, :], dim=-1) > 10)
        
        return body_contact

    def _get_forward_depth_obs(self, privileged = False):
        return torch.stack(self.sensor_tensor_dict["forward_depth"]).flatten(start_dim= 1)

    ##### The wrapper function to build and help processing observations #####
    def _get_obs_from_components(self, components: list, privileged = False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            # get the observation from specific component name
            obs.append(
                getattr(self, "_get_" + k + "_obs")(privileged) * \
                getattr(self.obs_scales, k, 1.)
            )
        obs = torch.cat(obs, dim= 1)
        return obs
    
    def _get_transition_from_components(self, components: list):
        transition_segments = self.get_transition_segment_from_components(components)
        transition = []
        for k, v in transition_segments.items():
            transition.append(
                getattr(self, "last_"+k)
            )
        for k, v in transition_segments.items():
            transition.append(
                getattr(self, k)
            )
        transition = torch.cat(transition, dim=1)
        return transition

    # defines observation segments, which tells the order of the entire flattened obs
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "proprio_with_goal" in components:
            segments["proprio_with_goal"] = (49,)
        if "height_measurements" in components:
            segments["height_measurements"] = (len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y),)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        if "robot_friction" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            segments["robot_friction"] = (1,)
        if "stumble_state" in components:
            num_dim = 0
            if "foot" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "hip" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "thigh" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "calf" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "base" in self.cfg.asset.collision_body_names:
                num_dim += 1
            segments["stumble_state"] = (num_dim,)
        return segments
    
    def get_estimated_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "lin_vel" in components:
            segments["lin_vel"] = (3,)
        if "height_measurements" in components:
            segments["height_measurements"] = (len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y),)
        if "robot_friction" in components:
            segments["robot_friction"] = (1,)
        if "stumble_state" in components:
            num_dim = 0
            if "foot" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "hip" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "thigh" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "calf" in self.cfg.asset.collision_body_names:
                num_dim += 4
            if "base" in self.cfg.asset.collision_body_names:
                num_dim += 1
            segments["stumble_state"] = (num_dim,)
        return segments
    
    def get_transition_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "dof_pos" in components:
            segments["dof_pos"] = (12,)
        if "dof_vel" in components:
            segments["dof_vel"] = (12,)
        if "base_height" in components:
            segments["base_height"] = (1,)
        if "base_lin_vel" in components:
            segments["base_lin_vel"] = (3,)
        if "base_ang_vel" in components:
            segments["base_ang_vel"] = (3,)
        if "commands_transition" in components:
            segments["commands_transition"] = (3,)
        return segments

    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
    
    def get_num_estimated_from_components(self, components):
        estimated_segments = self.get_estimated_segment_from_components(components)
        num_estimated = 0
        for k, v in estimated_segments.items():
            num_estimated += np.prod(v)
        return num_estimated
    
    def compute_observations(self):
        """ Computes observations
        """
        # force refresh graphics if needed
        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        if hasattr(self.cfg.env, "privileged_obs_components"):
            self.privileged_obs_buf = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", True),
            )
        else:
            self.privileged_obs_buf = None

        # wrap up to read the graphics data
        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break
        
        # add simple noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            
    def compute_transition_states(self):
        self.transition_buf = self._get_transition_from_components(
            self.cfg.env.transition_states_components,
        )
    
    # utility functions to meet old APIs and fit new obs logic
    @property
    def all_obs_components(self):
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def obs_segments(self):
        return self.get_obs_segment_from_components(self.cfg.env.obs_components)
    @property
    def privileged_obs_segments(self):
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_obs_segment_from_components(components)
    @property
    def num_obs(self):
        """ get this value from self.cfg.env """
        # assert "proprioception" in self.cfg.env.obs_components, "missing critical observation component 'proprioception'"
        return self.get_num_obs_from_components(self.cfg.env.obs_components)
    @num_obs.setter
    def num_obs(self, value):
        """ avoid setting self.num_obs """
        pass
    @property
    def num_privileged_obs(self):
        """ get this value from self.cfg.env """
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_num_obs_from_components(components)
    @property
    def num_estimated(self): # 3+(187)+others
        """ get this value from self.cfg.env """
        components = getattr(
            self.cfg.env,
            "estimated_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_num_estimated_from_components(components)
    @num_privileged_obs.setter
    def num_privileged_obs(self, value):
        """ avoid setting self.num_privileged_obs """
        pass
    #Done The wrapper function to build and help processing observations #####

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_terrain()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction for viewer
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if env_id == 0:
            all_obs_components = self.all_obs_components
            if "robot_config" in all_obs_components:
                all_obs_components
                self.robot_config_buffer = torch.empty(
                    self.num_envs, 1 + 3 + 1 + 12,
                    dtype= torch.float32,
                    device= self.device,
                )
        
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # allow config to override torque limits
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits, dtype= torch.float, device= self.device, requires_grad= False)
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if self.cfg.domain_rand.randomize_com:
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)
        
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # log max power across current env step
        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
        )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _init_goal_commands(self, env_ids):
        """ Randommly select goal commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new trajectory commands are needed
        """
        
        goal_x = torch_rand_float(-self.traj_command_ranges["goal_x"][1], self.traj_command_ranges["goal_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        goal_y = torch_rand_float(-self.traj_command_ranges["goal_y"][1], self.traj_command_ranges["goal_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # goal_z = torch_rand_float(self.traj_command_ranges["goal_z"][0], self.traj_command_ranges["goal_z"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        goal_yaw = torch_rand_float(self.traj_command_ranges["goal_yaw"][0], self.traj_command_ranges["goal_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        mask=(goal_x>-self.traj_command_ranges["goal_x"][0])&(goal_x<self.traj_command_ranges["goal_x"][0])&(goal_y>-self.traj_command_ranges["goal_y"][0])&(goal_y<self.traj_command_ranges["goal_y"][0])
        while(torch.any(mask)):
            mask_len=torch.sum(mask)
            goal_x[mask]=torch_rand_float(-self.traj_command_ranges["goal_x"][1], self.traj_command_ranges["goal_x"][1], (mask_len, 1), device=self.device).squeeze(1)
            goal_y[mask]=torch_rand_float(-self.traj_command_ranges["goal_y"][1], self.traj_command_ranges["goal_y"][1], (mask_len, 1), device=self.device).squeeze(1)
            mask=(goal_x>-self.traj_command_ranges["goal_x"][0])&(goal_x<self.traj_command_ranges["goal_x"][0])&(goal_y>-self.traj_command_ranges["goal_y"][0])&(goal_y<self.traj_command_ranges["goal_y"][0])
                
        self.goal_commands[env_ids, 0] = goal_x
        self.goal_commands[env_ids, 1] = goal_y
        self.goal_commands[env_ids, 2] = goal_yaw

        self.time_info[env_ids, 0] = self.cfg.env.episode_length_s
        self.time_info[env_ids, 1] = self.cfg.env.episode_length_s
        self.start_time[env_ids] = self.cfg.env.episode_length_s-torch_rand_float(self.traj_command_ranges["start_time"][0], self.traj_command_ranges["start_time"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if hasattr(self, "motor_strength"):
            actions = self.motor_strength * actions
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        if getattr(self.cfg.domain_rand, "init_dof_pos_ratio_range", None) is not None:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.init_dof_pos_ratio_range[0],
                self.cfg.domain_rand.init_dof_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        # self.dof_vel[env_ids] = 0. # history init method
        dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-3., 3.])
        self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)

        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        dof_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        dof_idx_int32 = dof_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.all_dof_states),
                                            gymtorch.unwrap_tensor(dof_idx_int32), len(dof_idx_int32))
        
    def _set_dofs(self, env_ids, dof_pos, dof_vel):
        """ sets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        dof_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        dof_idx_int32 = dof_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.all_dof_states),
                                            gymtorch.unwrap_tensor(dof_idx_int32), len(dof_idx_int32))
        
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "init_base_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(env_ids), 1), device=self.device)
                self.root_states[env_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(env_ids), 1), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation (roll, pitch, yaw)
        if hasattr(self.cfg.domain_rand, "init_base_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.init_base_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            if self.cfg.terrain.unify:
                base_yaw = torch_rand_float(
                    *([-0.2,0.2]),
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]*(self.root_states[env_ids, 1]<self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols))+\
                torch_rand_float(
                    *self.cfg.domain_rand.init_base_rot_range["yaw"],
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]*(self.root_states[env_ids, 1]>self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols))
            else:
                base_yaw = torch_rand_float(
                    *self.cfg.domain_rand.init_base_rot_range["yaw"],
                    (len(env_ids), 1),
                    device=self.device,
                )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
            self.root_states[env_ids, 3:7] = base_quat
        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        if isinstance(base_vel_range, (tuple, list)):
            if self.cfg.terrain.unify:
                self.root_states[env_ids, 7:13] = torch_rand_float(
                    *base_vel_range,
                    (len(env_ids), 6),
                    device=self.device,
                )*(self.root_states[env_ids, 1]>=self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols)).unsqueeze(1)+\
                torch_rand_float(
                    *([0.,0.]),
                    (len(env_ids), 6),
                    device=self.device,
                )*(self.root_states[env_ids, 1]<self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols)).unsqueeze(1)
                # [7:10]: lin vel, [10:13]: ang vel
            else:
                self.root_states[env_ids, 7:13] = torch_rand_float(
                    *base_vel_range,
                    (len(env_ids), 6),
                    device=self.device,
                )
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device,
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")
        
        # Each env has multiple actors. So the actor index is not the same as env_id. But robot actor is always the first.
        actor_idx = env_ids * self.all_root_states.shape[0] / self.num_envs
        actor_idx_int32 = actor_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.all_root_states),
                                                    gymtorch.unwrap_tensor(actor_idx_int32), len(actor_idx_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        if self.cfg.terrain.unify:
            self.root_states[:, 7:9]=torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)*\
            (self.root_states[:, 1]>self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols)).unsqueeze(1)+\
            self.root_states[:, 7:9]*(self.root_states[:, 1]<=self.cfg.terrain.terrain_width*(self.cfg.terrain.unify_cols)).unsqueeze(1)
        else:
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        move_up, move_down = self._get_terrain_curriculum_move(env_ids)
        self.terrain_time[env_ids] = 0
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def _get_terrain_types(self):
        return self.terrain_types

    def _get_terrain_curriculum_move(self, env_ids):
        goal_dis = torch.norm(self.delta_goal[env_ids, :2], dim=1)
        move_up = (goal_dis<0.5)
        move_down = (goal_dis>1.5)
        return move_up, move_down
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        obs_segments = self.get_obs_segment_from_components(cfg.env.obs_components)
        # write noise for each corresponding component.
        for k, v in obs_segments.items():
            segment_length = np.prod(v)
            # write sensor scale to provided noise_vec
            getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
            segment_start_idx += segment_length
        return noise_vec

    def _write_proprioception_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions

    def _write_proprio_with_goal_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        noise_vec[48:49] = 0. # time left

    def _write_height_measurements_noise(self, noise_vec):
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

    def _write_forward_depth_noise(self, noise_vec):
        noise_vec[:] = self.cfg.noise.noise_scales.forward_depth * self.cfg.noise.noise_level * self.obs_scales.forward_depth

    def _write_base_pose_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "base_pose"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.base_pose * self.cfg.noise.noise_level * self.obs_scales.base_pose
    
    def _write_robot_config_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_config"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_config * self.cfg.noise.noise_level * self.obs_scales.robot_config
        
    def _write_robot_friction_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_friction"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_friction * self.cfg.noise.noise_level * self.obs_scales.robot_friction

    def _write_stumble_state_noise(self, noise_vec):
        return

    #----------------------------------------
    
    def _generate_domain_rand_buffer(self):
        '''gererate domain rand buffer before init buffers
        '''
        
        # motor strengths and allow domain rand to change them
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_motor:
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, -1, 13)[:, 0, :] # (num_envs, 13)
        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, -1, 2)[:, :self.num_dof, :] # (num_envs, 2)
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., :self.num_dof, 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel, device=self.device)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)
        self.stuck_buf=torch.zeros(self.num_envs,dtype=torch.int,device=self.device)
        self.recover_buf=torch.zeros(self.num_envs,dtype=torch.int,device=self.device)
        self.terrain_time=torch.zeros(self.num_envs,dtype=torch.int,device=self.device)
        self.base_height=torch.zeros(self.num_envs,1,dtype=torch.float,device=self.device)
        self.last_base_height=torch.zeros_like(self.base_height)
        self.body_contact=torch.zeros(self.num_envs, len(self.collision_body_indices), dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.env.goal_command:
            self.goal_commands=torch.zeros(self.num_envs, 3, dtype=torch.float,device=self.device)
            self.delta_goal=torch.zeros(self.num_envs, 3, dtype=torch.float,device=self.device)
            self.time_info=torch.zeros(self.num_envs, 2, dtype=torch.float,device=self.device)
        self.start_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        # motor_strength
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if getattr(self.cfg.domain_rand, "randomize_motor", False):
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        # robot_config
        all_obs_components = self.all_obs_components
        self.robot_config_buffer = torch.empty(
            self.num_envs, 1 + 3 + 1 + self.num_actions,
            dtype= torch.float32,
            device= self.device,
        )
        assert len(self.envs) == len(self.actor_handles), "Number of envs and actor_handles must be the same. Other actor handles in the env must be put in npc_handles."
        for env_id, (env_h, actor_h) in enumerate(zip(self.envs, self.actor_handles)):
            actor_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_h, actor_h)
            actor_dof_props = self.gym.get_actor_dof_properties(env_h, actor_h)
            actor_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_h, actor_h)
            self.robot_config_buffer[env_id, 0] = actor_rigid_shape_props[0].friction
            self.robot_config_buffer[env_id, 1] = actor_rigid_body_props[0].com.x
            self.robot_config_buffer[env_id, 2] = actor_rigid_body_props[0].com.y
            self.robot_config_buffer[env_id, 3] = actor_rigid_body_props[0].com.z
            self.robot_config_buffer[env_id, 4] = actor_rigid_body_props[0].mass
        self.robot_config_buffer[:, 5:5+self.num_actions] = self.motor_strength

        # sensor tensors
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # add sensor dict, which will be filled during create sensor
        self.sensor_tensor_dict = defaultdict(list)

    def _reset_buffers(self, env_ids):
        if getattr(self.cfg.init_state, "zero_actions", False):
            self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_base_height[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_base_ang_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.stuck_buf[env_ids] = 0
        self.recover_buf[env_ids] = 0
        self.max_power_per_timestep[env_ids] = 0.
        if self.cfg.env.goal_command:
            self.delta_goal[env_ids] = 0.
        self.body_contact[env_ids] = 0.

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_sensors(self, env_handle= None, actor_handle= None):
        """ attach necessary sensors for each actor in each env
        Considering only one robot in each environment, this method takes only one actor_handle.
        Args:
            env_handle: env_handle from gym.create_env
            actor_handle: actor_handle from gym.create_actor
        Return:
            sensor_handle_dict: a dict of sensor_handles with key as sensor name (defined in cfg["sensor"])
        """
        return dict()
    
    def _create_npc(self, env_handle, env_idx):
        """ create additional opponent for each environment such as static objects, random agents
        or turbulance.
        """
        return dict()

    def _create_envs(self):
        """ Creates environments:
            1. loads the robot URDF/MJCF asset,
            2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
            3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        collision_body_names = []
        for name in self.cfg.asset.collision_body_names:
            collision_body_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        self._generate_domain_rand_buffer()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.npc_handles = [] # surrounding actors or objects or oppoents in each environment.
        self.sensor_handles = []
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
            npc_handle_dict = self._create_npc(env_handle, i)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.sensor_handles.append(sensor_handle_dict)
            self.npc_handles.append(npc_handle_dict)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.collision_body_indices = torch.zeros(len(collision_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(collision_body_names)):
            self.collision_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], collision_body_names[i])

    def _create_terrain(self):
        mesh_type = getattr(self.cfg.terrain, "mesh_type", None)
        if mesh_type=='plane':
            self._create_ground_plane()
        else:
            terrain_cls = self.cfg.terrain.selected
            self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs)
            self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
            self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if getattr(self.cfg.terrain, "mesh_type", None) is not None:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = copy(self.cfg.normalization.obs_scales)
        self.height_measurement_offset = copy(self.cfg.normalization.height_measurement_offset)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.env.goal_command:
            self.traj_command_ranges = class_to_dict(self.cfg.goal_command.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    ##### draw debug vis and the sub functions #####
    def _draw_measure_heights_vis(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 8, 8, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_goal_vis(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom1 = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(1, 1, 0))
        sphere_geom2 = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(0, 1, 1))
        for i in range(self.num_envs):
            goal = (self.goal_commands[i, :3]).cpu().numpy()
            x = goal[0] + self.env_origins[i, 0]
            y = goal[1] + self.env_origins[i, 1]
            z = 0
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom1, self.gym, self.viewer, self.envs[i], sphere_pose)
            x = self.env_origins[i, 0]
            y = self.env_origins[i, 1]
            z = 0
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom2, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_guide_position(self, target_position):
        self.gym.clear_lines(self.viewer)
        sphere_geom1 = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(1, 1, 0))
        sphere_geom2 = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(0, 1, 1))
        for i in range(self.num_envs):
            goal = (self.root_states[i, 0:2]).cpu().numpy()
            x = 2.5
            y = min(goal[1]+3,target_position[1])
            z = 0
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom1, self.gym, self.viewer, self.envs[i], sphere_pose)
            x = 2.5
            y = target_position[1]
            z = 0
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom2, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _draw_sensor_vis(self, env_h, sensor_hd):
        for sensor_name, sensor_h in sensor_hd.items():
            if "camera" in sensor_name:
                camera_transform = self.gym.get_camera_transform(self.sim, env_h, sensor_h)
                cam_axes = gymutil.AxesGeometry(scale= 0.1)
                gymutil.draw_lines(cam_axes, self.gym, self.viewer, env_h, camera_transform)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # clear all drawings first
        self.gym.clear_lines(self.viewer)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _fill_extras(self, env_ids):
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_frame_' + key] = torch.nanmean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            if len(env_ids) > 0:
                self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels[env_ids].float())
                self.extras["episode"]["terrain_level_min"] = torch.min(self.terrain_levels[env_ids].float())
        # log power related info
        self.extras["episode"]["max_power_throughout_episode"] = self.max_power_per_timestep[env_ids].max().cpu().item()
        # log running range info
        pos_x = self.root_states[env_ids][:, 0] - self.env_origins[env_ids][:, 0]
        pos_y = self.root_states[env_ids][:, 1] - self.env_origins[env_ids][:, 1]
        self.extras["episode"]["max_pos_x"] = torch.max(pos_x).cpu()
        self.extras["episode"]["min_pos_x"] = torch.min(pos_x).cpu()
        self.extras["episode"]["max_pos_y"] = torch.max(pos_y).cpu()
        self.extras["episode"]["min_pos_y"] = torch.min(pos_y).cpu()

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = self.time_out_buf.float().sum() / self.reset_buf.float().sum()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_orientation_y(self):
        # Penalize non flat base orientation
        return torch.square(self.projected_gravity[:, 1])

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_rotation(self):
        # Penalize rotation
        return torch.square(self.base_ang_vel[:, 2])
    
    def _reward_y(self):
        return torch.square(self.root_states[:,1]-self.env_origins[:,1])
    
    def _reward_turn_around(self):
        r, p, y = get_euler_xyz(self.base_quat)
        return torch.mean(torch.abs(y))
    
    def _reward_foot_slip(self):
        # penalize foot slip
        self.feet_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :]  # env_num*4*13
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        slip_vel=torch.zeros([self.num_envs,4,3],device=self.device)
        
        for i in range(4):
            self.feet_states_rot=quat_rotate_inverse(self.base_quat,self.feet_states[:,i,7:10])
            slip_vel[:,i,:]=contact_filt[:,i].unsqueeze(1)*self.feet_states_rot
            
        # slip_vel=slip_vel.reshape(self.num_envs,-1)
        slip_vel=torch.sum(torch.norm(slip_vel,dim=2),dim=1)
        return slip_vel
    
    def _reward_feet_air_time_l1(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)+10*torch.sum(torch.clip((0.5-self.feet_air_time) * first_contact,max=0),dim=1)
        self.feet_air_time *= ~contact_filt

        return rew_airTime
    
    def _reward_foot_contact(self):
        # penalize foot slip
        forces = self.contact_forces[:, self.feet_indices, 2]
        forces_reward = torch.square(forces[:,0]+forces[:,2]-forces[:,1]-forces[:,3])
        return forces_reward
    
    def _reward_get_goal(self):
        goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
        time = self.time_info[:, 1]
        total_time = self.time_info[:, 0]
        reward_time = self.cfg.rewards.time_goal_reward*total_time
        rew = 1 / (0.4 + torch.clip(goal_dis, min=0.))
        vel_now=torch.norm(self.base_lin_vel[:, :2], dim=1)
        rew = rew *(torch.abs(self.base_ang_vel[:, 2])<1.)*(vel_now<self.cfg.rewards.velocity_limit)
        return rew
    
    def _reward_reach_goal(self):
        goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
        rew = (goal_dis<0.2)
        return rew
    
    def _reward_goal_yaw(self):
        yaw_dis = torch.norm(self.delta_goal[:, 2:], dim=1)
        time = self.time_info[:, 1]
        total_time = self.time_info[:, 0]
        reward_time = self.cfg.rewards.time_goal_reward*total_time
        rew = 1 / (0.4 + torch.clip((yaw_dis-0.2), min=0.))
        rew = rew * (time<reward_time)*(torch.abs(self.base_ang_vel[:, 2]<1))
        return rew
    
    def _reward_goal_heading(self):
        goal_dis=torch.norm(self.delta_goal[:, :2], dim=1).unsqueeze(-1)
        goal_vec=(self.delta_goal[:, :2])/(goal_dis+0.00001)
        rew = goal_vec[:, 0] + (goal_dis.squeeze(-1)==0)
        return rew
    
    def _reward_stand_still_at_finish(self):
        goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (goal_dis < 0.2)
    
    def _reward_velocity_at_finish(self):
        goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
        return torch.abs(self.base_ang_vel[:, 2]) * (torch.norm(self.base_lin_vel[:, :2], dim=1))* (goal_dis < 0.2)
    
    def _reward_stall(self):
        goal_dis = torch.norm(self.delta_goal[:, :2], dim=1)
        return 1.*(torch.norm(self.base_lin_vel[:, :2], dim = 1) < 0.1)* (goal_dis > 0.25)
    
    def _reward_vel_safe(self):
        vel_now=torch.norm(self.base_lin_vel[:, :2], dim=1)
        rew = (torch.abs(self.base_ang_vel[:, 2])<1.)*(vel_now<self.cfg.rewards.velocity_limit)
        return rew
    
    def _reward_ang_vel_safe(self):
        rew = (torch.abs(self.base_ang_vel[:, 2])<self.cfg.rewards.velocity_limit)
        return rew