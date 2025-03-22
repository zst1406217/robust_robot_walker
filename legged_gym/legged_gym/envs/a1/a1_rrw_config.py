import numpy as np
import os.path as osp
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO

class A1RRWCfg( A1RoughCfg ):
    class env( A1RoughCfg.env ):
        num_envs = 4096 # 8192
        obs_components = [
            "proprio_with_goal", # 49
            # "height_measurements", # 187
            "base_pose",
            "robot_config",
        ]
        goal_command = False

    class sensor:
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link
    
        class proprioception:
            delay_action_obs = False
            latency_range = [0.0, 0.0]
            latency_resample_time = 2.0 # [s]
            
    class init_state( A1RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
    
    class terrain( A1RoughCfg.terrain ):
        mesh_type = "trimesh" # Don't change
        num_rows = 20
        num_cols = 50
        selected = "TerrainPerlin"
        max_init_terrain_level = 0
        border_size = 0
        slope_treshold = 20.
        unify = False

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]

        TerrainPerlin_kwargs = dict(
            zScale= 0.15,
            frequency= 10,
        )
    
    class commands( A1RoughCfg.commands ):
        heading_command = False
        resampling_time = 10 # [s]
        class ranges( A1RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control( A1RoughCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 1.}
        action_scale = 0.5
        torque_limits = 25 # override the urdf
        computer_clip_torque = False
        motor_clip_torque = False

    class asset( A1RoughCfg.asset ):
        penalize_contacts_on = ['base', 'hip', 'thigh', 'calf', 'foot']
        terminate_after_contacts_on = ["base", "imu"]
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]
        collision_body_names = ['base', 'thigh', 'calf', 'foot']

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
        ]

        roll_kwargs = dict(
            threshold= 1.0, # [rad]
            tilt_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad]
            jump_threshold= 1.6,
            leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.15, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = False

    class domain_rand( A1RoughCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]

        randomize_friction = True
        friction_range = [0., 2.]

        init_base_pos_range = dict(
            x= [-0.2, 0.2],
            y= [-0.2, 0.2],
        )

        push_robots = False 

    class rewards( A1RoughCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -2e-5
            legs_energy = -0.
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
        soft_dof_pos_limit = 0.01

    class normalization( A1RoughCfg.normalization ):
        class obs_scales( A1RoughCfg.normalization.obs_scales ):
            forward_depth = 1.
            base_pose = [0., 0., 0., 1., 1., 1.]
            engaging_block = 1.
            robot_config = 1.
            robot_friction = 1.
        height_measurement_offset = -0.3

    class noise( A1RoughCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales( A1RoughCfg.noise.noise_scales ):
            forward_depth = 0.1
            base_pose = 1.0

    class viewer( A1RoughCfg.viewer ):
        pos = [0, 0, 5]  # [m]
        lookat = [5., 5., 2.]  # [m]

        draw_volume_sample_points = False

    class sim( A1RoughCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.061, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

    class curriculum:
        no_moveup_when_fall = False

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1RRWCfgPPO( A1RoughCfgPPO ):
    class algorithm( A1RoughCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 1e-12

    class policy( A1RoughCfgPPO.policy ):
        rnn_type = 'gru'
        mu_activation = "tanh"
        activation = 'elu'
    
    class runner( A1RoughCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        experiment_name = "rrw_a1"
        resume = False
        max_iterations = 10000
        save_interval = 500