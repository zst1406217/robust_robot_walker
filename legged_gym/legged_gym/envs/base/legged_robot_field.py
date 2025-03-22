from collections import OrderedDict, defaultdict
import itertools
import numpy as np
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.observation import get_obs_slice
from .legged_robot_config import LeggedRobotCfg

class LeggedRobotField(LeggedRobot):
    """ NOTE: Most of this class implementation does not depend on the terrain. Check where
    `check_BarrierTrack_terrain` is called to remove the dependency of BarrierTrack terrain.
    """
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotField.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        cfg.terrain.measure_heights = True # force height measurement that have full obs from parent class implementation.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    def check_BarrierTrack_terrain(self):

        return True

    ##### Working on simulation steps #####
    def pre_physics_step(self, actions):
        self.volume_sample_points_refreshed = False        
        return super().pre_physics_step(actions)
    
    def check_termination(self):
        return_ = super().check_termination()
        if not hasattr(self.cfg, "termination"): return return_
        
        r, p, y = get_euler_xyz(self.base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        z = self.root_states[:, 2] - self.env_origins[:, 2]
        
        if "roll" in self.cfg.termination.termination_terms:
            r_term_buff = torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]
            self.reset_buf |= r_term_buff
        if "pitch" in self.cfg.termination.termination_terms:
            p_term_buff = torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]
            self.reset_buf |= p_term_buff
        if "z_low" in self.cfg.termination.termination_terms:
            z_low_term_buff = z < self.cfg.termination.z_low_kwargs["threshold"]
            self.reset_buf |= z_low_term_buff
        if "z_high" in self.cfg.termination.termination_terms:
            z_high_term_buff = z > self.cfg.termination.z_high_kwargs["threshold"]
            self.reset_buf |= z_high_term_buff

        if getattr(self.cfg.termination, "timeout_at_border", False):
            track_idx = self.terrain.get_track_idx(self.root_states[:, :3], clipped= False)
            # The robot is going +x direction, so no checking for row_idx <= 0
            out_of_border_buff = track_idx[:, 0] >= self.terrain.cfg.num_rows
            out_of_border_buff |= track_idx[:, 1] < 0
            out_of_border_buff |= track_idx[:, 1] >= self.terrain.cfg.num_cols
            self.time_out_buf |= out_of_border_buff
            self.reset_buf |= out_of_border_buff
        if getattr(self.cfg.termination, "timeout_at_finished", False):
            x = self.root_states[:, 0] - self.env_origins[:, 0]
            y = self.root_states[:, 1] - self.env_origins[:, 1]
            finished_buffer_x = torch.abs(x) > (self.cfg.terrain.terrain_width/2-0.7)
            finished_buffer_y = torch.abs(y) > (self.cfg.terrain.terrain_length/2-0.7)
            finished_buffer=finished_buffer_x|finished_buffer_y
            self.time_out_buf |= finished_buffer
            self.reset_buf |= finished_buffer
        
        return return_

    ##### Dealing with observations #####
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        
        super()._init_buffers()

        # projected gravity bias (if needed)
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            print("Initializing gravity bias for domain randomization")
            # add cross trajectory domain randomization on projected gravity bias
            # uniform sample from range
            self.gravity_bias = torch.rand(self.num_envs, 3, dtype= torch.float, device= self.device, requires_grad= False)
            self.gravity_bias[:, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[:, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "velocity_sample_points"): self.velocity_sample_points[env_ids] = 0.

        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            assert hasattr(self, "gravity_bias")
            self.gravity_bias[env_ids] = torch.rand_like(self.gravity_bias[env_ids])
            self.gravity_bias[env_ids, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[env_ids, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]
        
        return return_
        
    def _prepare_reward_function(self):
        return_ = super()._prepare_reward_function()

        return return_

    def _get_proprioception_obs(self, privileged= False):
        obs_buf = super()._get_proprioception_obs(privileged= privileged)
        
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False) and (not privileged):
            assert hasattr(self, "gravity_bias")
            proprioception_slice = get_obs_slice(self.obs_segments, "proprioception")
            obs_buf[:, proprioception_slice[0].start + 6: proprioception_slice[0].start + 9] += self.gravity_bias
        
        return obs_buf

    ##### adds-on with building the environment #####
    def _create_envs(self):        
        return_ = super()._create_envs()
        
        front_hip_names = getattr(self.cfg.asset, "front_hip_names", ["FR_hip_joint", "FL_hip_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_hip_names = getattr(self.cfg.asset, "rear_hip_names", ["RR_hip_joint", "RL_hip_joint"])
        self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_hip_names):
            self.rear_hip_indices[i] = self.dof_names.index(name)

        hip_names = front_hip_names + rear_hip_names
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
            
        front_thigh_names = getattr(self.cfg.asset, "front_thigh_names", ["FR_thigh_joint", "FL_thigh_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_thigh_names = getattr(self.cfg.asset, "rear_thigh_names", ["RR_thigh_joint", "RL_thigh_joint"])
        self.rear_thigh_indices = torch.zeros(len(rear_thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_thigh_names):
            self.rear_thigh_indices[i] = self.dof_names.index(name)

        thigh_names = front_thigh_names + rear_thigh_names
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        
        return return_

    ##### Additional rewards #####
    def _reward_lin_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim= 1)

    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)

    def _reward_tracking_world_vel(self):
        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_legs_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_legs_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_alive(self):
        return 1.

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_lin_cmd(self):
        """ This reward term does not depend on the policy, depends on the command """
        return torch.norm(self.commands[:, :2], dim= 1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]
    
    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])
    
    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])
    
    def _reward_lin_vel_z_square(self):
        return torch.square(self.root_states[:, 9])

    def _reward_lin_pos_y(self):
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])
    
    def _reward_yaw_abs(self):
        """ Aiming for the robot yaw to be zero (pointing to the positive x-axis) """
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw < -np.pi] += np.pi * 2 # to range (-pi, pi)
        return torch.abs(yaw)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_front_hip_pos(self):
        """ Reward the robot to stop moving its front hips """
        return torch.sum(torch.square(self.dof_pos[:, self.front_hip_indices] - self.default_dof_pos[:, self.front_hip_indices]), dim=1)

    def _reward_rear_hip_pos(self):
        """ Reward the robot to stop moving its rear hips """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_hip_indices] - self.default_dof_pos[:, self.rear_hip_indices]), dim=1)
    
    def _reward_thigh_acc(self):
        """ Reward the robot to stop moving its rear thighs """
        return torch.sum(torch.square((self.last_dof_vel[:, self.thigh_indices] - self.dof_vel[:, self.thigh_indices]) / self.dt), dim=1)