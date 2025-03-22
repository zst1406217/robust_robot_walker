import random

from isaacgym.torch_utils import torch_rand_float
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class LeggedRobotNoisyMixin:
    """ This class should be independent from the terrain, but depend on the sensors of the parent
    class.
    """

    def clip_position_action_by_torque_limit(self, actions_scaled):
        """ For position control, scaled actions should be in the coordinate of robot default dof pos
        """
        if hasattr(self, "proprioception_output"):
            dof_vel = self.proprioception_output[:, -24:-12] / self.obs_scales.dof_vel
            dof_pos_ = self.proprioception_output[:, -36:-24] / self.obs_scales.dof_pos
        else:
            dof_vel = self.dof_vel
            dof_pos_ = self.dof_pos - self.default_dof_pos
        p_limits_low = (-self.torque_limits) + self.d_gains*dof_vel
        p_limits_high = (self.torque_limits) + self.d_gains*dof_vel
        actions_low = (p_limits_low/self.p_gains) + dof_pos_
        actions_high = (p_limits_high/self.p_gains) + dof_pos_
        actions_scaled_torque_clipped = torch.clip(actions_scaled, actions_low, actions_high)
        return actions_scaled_torque_clipped

    def pre_physics_step(self, actions):
        self.proprioception_refreshed = False
        return_ = super().pre_physics_step(actions)

        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        if getattr(self.cfg.control, "computer_clip_torque", False):
            self.actions_scaled = self.actions * self.cfg.control.action_scale
            control_type = self.cfg.control.control_type
            if control_type == "P":
                actions_scaled_torque_clipped = self.clip_position_action_by_torque_limit(self.actions_scaled)
            else:
                raise NotImplementedError
        else:
            actions_scaled_torque_clipped = self.actions * self.cfg.control.action_scale

        if getattr(self.cfg.control, "action_delay", False):
            # always put the latest action at the end of the buffer
            self.actions_history_buffer = torch.roll(self.actions_history_buffer, shifts= -1, dims= 0)
            self.actions_history_buffer[-1] = actions_scaled_torque_clipped
            # get the delayed action
            self.action_delayed_frames = ((self.current_action_delay / self.dt) + 1).to(int)
            self.actions_scaled_torque_clipped = self.actions_history_buffer[
                -self.action_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ]
        else:
            self.actions_scaled_torque_clipped = actions_scaled_torque_clipped
        
        return return_
    
    def _compute_torques(self, actions):
        """ The input actions will not be used, instead the scaled clipped actions will be used.
        Please check the computation logic whenever you change anything.
        """
        if not hasattr(self.cfg.control, "motor_clip_torque"):
            return super()._compute_torques(actions)
        else:
            if hasattr(self, "motor_strength"):
                actions_scaled_torque_clipped = self.motor_strength * self.actions_scaled_torque_clipped
            else:
                actions_scaled_torque_clipped = self.actions_scaled_torque_clipped
            control_type = self.cfg.control.control_type
            if control_type == "P":
                torques = self.p_gains * (actions_scaled_torque_clipped + self.default_dof_pos - self.dof_pos) \
                    - self.d_gains * self.dof_vel
            else:
                raise NotImplementedError
            if self.cfg.control.motor_clip_torque:
                torques = torch.clip(
                    torques,
                    -self.torque_limits * self.cfg.control.motor_clip_torque,
                    self.torque_limits * self.cfg.control.motor_clip_torque,
                )
            return torques
        
    def post_decimation_step(self, dec_i):
        return_ = super().post_decimation_step(dec_i)
        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        ### The set torque limit is usally smaller than the robot dataset
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        
        return return_
    
    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)
        
        self.extras["episode"]["max_torques"] = self.max_torques[env_ids]
        self.max_torques[env_ids] = 0.
        self.extras["episode"]["torque_exceed_count_substeps_per_envstep"] = self.torque_exceed_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.torque_exceed_count_substep[env_ids] = 0
        self.extras["episode"]["torque_exceed_count_envstep"] = self.torque_exceed_count_envstep[env_ids]
        self.torque_exceed_count_envstep[env_ids] = 0
        self.extras["episode"]["out_of_dof_pos_limit_count_substep"] = self.out_of_dof_pos_limit_count_substep[env_ids] / self.episode_length_buf[env_ids]
        self.out_of_dof_pos_limit_count_substep[env_ids] = 0

        return return_
    
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        if hasattr(self, "actions_history_buffer"):
            resampling_time = getattr(self.cfg.control, "action_delay_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_action_delay(resample_env_ids)

        if hasattr(self, "proprioception_buffer"):
            resampling_time = getattr(self.cfg.sensor.proprioception, "latency_resampling_time", self.dt)
            resample_env_ids = (self.episode_length_buf % int(resampling_time / self.dt) == 0).nonzero(as_tuple= False).flatten()
            if len(resample_env_ids) > 0:
                self._resample_proprioception_latency(resample_env_ids)

        self.torque_exceed_count_envstep[(torch.abs(self.substep_torques) > self.torque_limits).any(dim= 1).any(dim= 1)] += 1
        
    def _resample_action_delay(self, env_ids):
        self.current_action_delay[env_ids] = torch_rand_float(
            self.cfg.control.action_delay_range[0],
            self.cfg.control.action_delay_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()
    
    def _resample_proprioception_latency(self, env_ids):
        self.current_proprioception_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.proprioception.latency_range[0],
            self.cfg.sensor.proprioception.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _resample_forward_camera_latency(self, env_ids):
        self.current_forward_camera_latency[env_ids] = torch_rand_float(
            self.cfg.sensor.forward_camera.latency_range[0],
            self.cfg.sensor.forward_camera.latency_range[1],
            (len(env_ids), 1),
            device= self.device,
        ).flatten()

    def _init_buffers(self):
        return_ = super()._init_buffers()
        all_obs_components = self.all_obs_components

        if getattr(self.cfg.control, "action_delay", False):
            assert hasattr(self.cfg.control, "action_delay_range") and hasattr(self.cfg.control, "action_delay_resample_time"), "Please specify action_delay_range and action_delay_resample_time in the config file."
            """ Used in pre-physics step """
            self.cfg.control.action_history_buffer_length = int((self.cfg.control.action_delay_range[1] + self.dt) / self.dt)
            self.actions_history_buffer = torch.zeros(
                (
                    self.cfg.control.action_history_buffer_length,
                    self.num_envs,
                    self.num_actions,
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_action_delay = torch_rand_float(
                self.cfg.control.action_delay_range[0],
                self.cfg.control.action_delay_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.action_delayed_frames = ((self.current_action_delay / self.dt) + 1).to(int)

        if "proprioception" in all_obs_components and hasattr(self.cfg.sensor, "proprioception"):
            """ Adding proprioception delay buffer """
            self.cfg.sensor.proprioception.buffer_length = int((self.cfg.sensor.proprioception.latency_range[1] + self.dt) / self.dt)
            self.proprioception_buffer = torch.zeros(
                (
                    self.cfg.sensor.proprioception.buffer_length,
                    self.num_envs,
                    self.get_num_obs_from_components(["proprioception"]),
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_proprioception_latency = torch_rand_float(
                self.cfg.sensor.proprioception.latency_range[0],
                self.cfg.sensor.proprioception.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)

        if "proprio_with_goal" in all_obs_components and hasattr(self.cfg.sensor, "proprioception"):
            """ Adding proprioception delay buffer """
            self.cfg.sensor.proprioception.buffer_length = int((self.cfg.sensor.proprioception.latency_range[1] + self.dt) / self.dt)
            self.proprioception_buffer = torch.zeros(
                (
                    self.cfg.sensor.proprioception.buffer_length,
                    self.num_envs,
                    self.get_num_obs_from_components(["proprio_with_goal"]),
                ),
                dtype= torch.float32,
                device= self.device,
            )
            self.current_proprioception_latency = torch_rand_float(
                self.cfg.sensor.proprioception.latency_range[0],
                self.cfg.sensor.proprioception.latency_range[1],
                (self.num_envs, 1),
                device= self.device,
            ).flatten()
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)

        self.contour_detection_kernel = torch.zeros(
            (8, 1, 3, 3),
            dtype= torch.float32,
            device= self.device,
        )
        # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
        return return_

    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        if hasattr(self, "actions_history_buffer"):
            self.actions_history_buffer[:, env_ids] = 0.
            self.action_delayed_frames[env_ids] = self.cfg.control.action_history_buffer_length
        if hasattr(self, "proprioception_buffer"):
            self.proprioception_buffer[:, env_ids] = 0.
            self.proprioception_delayed_frames[env_ids] = self.cfg.sensor.proprioception.buffer_length
        return return_


    def _get_proprioception_obs(self, privileged= False):
        if not self.proprioception_refreshed and hasattr(self.cfg.sensor, "proprioception") and (not privileged):
            self.proprioception_buffer = torch.cat([
                self.proprioception_buffer[1:],
                super()._get_proprioception_obs().unsqueeze(0),
            ], dim= 0)
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used. [0.04-0.0075, 0.04+0.0025]
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)
            self.proprioception_output = self.proprioception_buffer[
                -self.proprioception_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            ### NOTE: WARN: ERROR: remove this code in final version, no action delay should be used.
            if getattr(self.cfg.sensor.proprioception, "delay_action_obs", False) or getattr(self.cfg.sensor.proprioception, "delay_privileged_action_obs", False):
                raise ValueError("LeggedRobotNoisy: No action delay should be used. Please remove these settings")
            # The last-action is not delayed.
            self.proprioception_output[:, -12:] = self.proprioception_buffer[-1, :, -12:]
            self.proprioception_refreshed = True
        if not hasattr(self.cfg.sensor, "proprioception") or privileged:
            return super()._get_proprioception_obs(privileged)

        return self.proprioception_output.flatten(start_dim= 1)
    
    def _get_proprio_with_goal_obs(self, privileged= False):
        if not self.proprioception_refreshed and hasattr(self.cfg.sensor, "proprioception") and (not privileged):
            self.proprioception_buffer = torch.cat([
                self.proprioception_buffer[1:],
                super()._get_proprio_with_goal_obs().unsqueeze(0),
            ], dim= 0)
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used. [0.04-0.0075, 0.04+0.0025]
            self.proprioception_delayed_frames = ((self.current_proprioception_latency / self.dt) + 1).to(int)
            self.proprioception_output = self.proprioception_buffer[
                -self.proprioception_delayed_frames,
                torch.arange(self.num_envs, device= self.device),
            ].clone()
            ### NOTE: WARN: ERROR: remove this code in final version, no action delay should be used.
            if getattr(self.cfg.sensor.proprioception, "delay_action_obs", False) or getattr(self.cfg.sensor.proprioception, "delay_privileged_action_obs", False):
                raise ValueError("LeggedRobotNoisy: No action delay should be used. Please remove these settings")
            # The last-action and time_info is not delayed.
            self.proprioception_output[:, -13:] = self.proprioception_buffer[-1, :, -13:]
            self.proprioception_refreshed = True
        if not hasattr(self.cfg.sensor, "proprioception") or privileged:
            return super()._get_proprio_with_goal_obs(privileged)

        return self.proprioception_output.flatten(start_dim= 1)
    
    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > self.torque_limits
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)
    
    def _reward_exceed_torque_limits_square(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.square(exceeded_torques).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)
