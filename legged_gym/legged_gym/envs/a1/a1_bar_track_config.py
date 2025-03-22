import os.path as osp
from legged_gym.envs.a1.a1_rrw_config import A1RRWCfg, A1RRWCfgPPO

class A1BarTrackCfg( A1RRWCfg ):
    class env( A1RRWCfg.env ):
        num_envs = 4096
        obs_components = [
            "proprio_with_goal", # 49
            "robot_friction",
            "stumble_state",
        ]
        privileged_obs_components = [
            "proprio_with_goal",
            "robot_friction", 
            "stumble_state",
        ]
        estimated_obs_components = [ # The first one must be lin_vel
            "lin_vel",
            "robot_friction", 
            "stumble_state",
        ]
        transition_states_components = [
            "dof_pos",
            "base_height",
            "base_lin_vel",
            "base_ang_vel", 
        ]
        use_lin_vel = True
        privileged_use_lin_vel = True
        goal_command = True
        
    class init_state( A1RRWCfg.init_state ):
        zero_actions=True

    class goal_command:
        class ranges:
            goal_x = [0, 3.5]
            goal_y = [0., 3.5]
            goal_yaw = [-3.14, 3.14]
            start_time = [0., 0]

    class terrain( A1RRWCfg.terrain ):
        num_rows = 1
        num_cols = 1
        single_terrain = False
        num_bars = 10
        num_pits = 10
        num_poles = 50
        terrain_length = 60.
        terrain_width = 5.
        min_bar_width = 0.01
        max_bar_width = 0.05
        min_bar_height = 0.05
        max_bar_height = 0.20
        min_pit_width = 0.05
        max_pit_width = 0.2

        selected = "TerrainStumbleBarTrack"
        TerrainPerlin_kwargs = dict(
            zScale= 0.1,
            frequency= 10,
        )
        unify = True
        stair_cols=8
        max_init_terrain_level = 1
        curriculum = True
        measure_heights = True

        min_zScale=0.
        max_zScale=0.1

        pit_length=2
        pit_width=0.05
        min_num_pit=0
        max_num_pit=0

        min_num_pole=0
        max_num_pole=0

        min_num_bar=5
        max_num_bar=10

        platform_size=1.0
        border_width=0.5

        slope_treshold = 2

    class commands( A1RRWCfg.commands ):
        class ranges( A1RRWCfg.commands.ranges ):
            lin_vel_x = [-1., 1.]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.] 
        resampling_time = 15.

    class asset( A1RRWCfg.asset ):
        collision_body_names = ['base', 'hip', 'thigh', 'foot', 'calf']
        step_num=20

    class domain_rand( A1RRWCfg.domain_rand ):
        class com_range( A1RRWCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]
        max_push_vel_ang = 0.5
        push_robots = True
        push_interval_s = 9
        init_base_vel_range = [-1.,1.]
        init_base_rot_range = dict(
            roll= [0, 0],
            pitch= [0, 0],
            yaw= [-3.14,3.14]
        )

    class rewards( A1RRWCfg.rewards ):
        class scales:
            # task reward
            get_goal = 5.0
            stand_still_at_finish = -1.
            velocity_at_finish = -1.
            stall = -2
            goal_heading = 3.
            vel_safe = 2.0
            # smooth reward
            legs_energy_substeps = -1e-6
            alive = 3
            exceed_dof_pos_limits = -0.02
            exceed_torque_limits_l1norm = -4.0
            orientation = -0.5
            feet_contact_forces = -5e-3
            termination = -20
            feet_air_time_l1 = 0.1
            dof_vel = -0.002
            dof_acc = -2e-6
            front_hip_pos = -0.1
            rear_hip_pos = -0.1
            foot_contact = -2e-5
            ang_vel_xy = -0.2
        
        soft_dof_pos_limit = 0.9
        max_contact_force = 60.0
        tracking_sigma = 0.25
        base_height_target = 0.32
        time_goal_reward = 0.8
        velocity_limit = 0.8
        
    class termination(A1RRWCfg.termination):
        termination_terms = [
            "roll",
            "pitch",
        ]
        stuck_kwargs = dict(
            threshold= 0.1,
            steps=20,
            recover_steps=50
        )
        # additional factors that determines whether to terminates the episode
        timeout_at_finished = True
        
    class noise( A1RRWCfg.noise ):
        add_noise = True # disable internal uniform +- 1 noise, and no noise in proprioception
        class noise_scales( A1RRWCfg.noise.noise_scales ):
            lin_vel = 0.05
            robot_config = 0.
            height_measurements = 0.
        
    class sensor( A1RRWCfg.sensor ):
        class proprioception:
            delay_action_obs = False
            latency_range = [0.04-0.0025, 0.04+0.0075] # [min, max] in seconds
            latency_resample_time = 2.0 # [s]    
    

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class A1BarTrackCfgPPO( A1RRWCfgPPO ):
    class policy(A1RRWCfgPPO.policy):
        scan_encoder_dims = None
        stumble_encoder_dims = [32, 16, 4]
        rnn_type = "lstm"
        has_scan = True if "height_measurements" in A1BarTrackCfg.env.obs_components else False
    
    class algorithm( A1RRWCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.05

    class runner( A1RRWCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = "PPO"
        num_steps_per_env = 50
        amp_reward_coeff = 0.1
        resume = True
        load_run = ""

        run_name = "".join(["WalkByBarTrack",
        ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])
        save_interval=2000
        max_iterations = 40000
        
    class estimator:
        train_with_estimated_states = True
        train_together = True
        learning_rate = 1.e-4
        input_dim = 46
        rnn_type = 'lstm'
        rnn_hidden_size = 256
        latent_encoder_hidden_dims=[256, 128]
        estimator_decay = 0.
        decay_start_step = 0
        ppo_train_together = True
        estimator_loss_coeff = 0.
        class_loss_coeff = 0.
    
    class discriminator:
        state_size=19
        hidden_sizes=[1024, 512]
        learning_rate = 1.e-4
        file_path = './mpc_data_no_command.npy'
        batch_size = 1024
        gradient_penalty = 10