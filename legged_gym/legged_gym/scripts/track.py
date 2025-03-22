import numpy as np
from isaacgym import gymtorch, gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym.torch_utils import get_euler_xyz
import torch
from tqdm import tqdm
import os

def create_recording_camera(gym, env_handle,
        resolution= (1920*5*2, 1080*4),
        h_fov= 86,
        actor_to_attach= None,
        transform= None, # related to actor_to_attach
    ):
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.width = resolution[0]
    camera_props.height = resolution[1]
    camera_props.horizontal_fov = h_fov
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    if actor_to_attach is not None:
        gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_to_attach,
            transform,
            gymapi.FOLLOW_POSITION,
        )
    elif transform is not None:
        gym.set_camera_transform(
            camera_handle,
            env_handle,
            transform,
        )
    return camera_handle

@ torch.no_grad()
def track(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    env_len = env_cfg.terrain.terrain_length

    train_cfg.runner.resume = True
    env_cfg.terrain.curriculum = False
    env_cfg.env.num_envs = 1000
    env_cfg.commands.heading_command = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.unify=False

    env_cfg.domain_rand.init_dof_pos_ratio_range = [1.0, 1.0]
    env_cfg.domain_rand.init_base_vel_range = [0., 0.]
    env_cfg.domain_rand.init_base_rot_range = dict(
            roll= [0, 0],
            pitch= [0, 0],
            yaw= [3.14 / 2 ,3.14 / 2 ]
        )
    env_cfg.domain_rand.init_base_pos_range = dict(
            x= [0, 0],
            y= [- env_len / 2 + 1.5, - env_len / 2 + 1.5],
        )
    
    
    env_cfg.viewer.debug_viz = True
    env_cfg.viewer.draw_volume_sample_points = False

    env_cfg.commands.resampling_time=int(1e16)
    env_cfg.env.episode_length_s=int(1e16)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()
    print(train_cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg= False,
    )

    agent_model = ppo_runner.alg.actor_critic
    policy = ppo_runner.alg.act_play

    # env_cfg.viewer.pos = [4., -6., 1]
    env_cfg.viewer.pos = [2.5, 30, 35]
    # env_cfg.viewer.pos = [0., 2., 5]
    env_cfg.viewer.lookat = [2.,30., 0]

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.6, 0., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    camera_follow_id = 0 # only effective when CAMERA_FOLLOW
    env.set_camera(camera_position, camera_position + camera_direction)

    target_position = np.array([env_cfg.terrain.terrain_width / 2, env_len -1])  # set target
    success_count = 0
    total_runs = 15000
    terminated = torch.zeros(env_cfg.env.num_envs, dtype=torch.bool).cpu().numpy()
    finish_time = torch.zeros(env_cfg.env.num_envs).cpu().numpy()
    travel_distance = torch.zeros(env_cfg.env.num_envs).cpu().numpy()
    num_terminated = 0

    if RECORD_FRAMES:
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(env_cfg.viewer.pos[0]+5, env_cfg.viewer.pos[1], env_cfg.viewer.pos[2])
        transform.r = gymapi.Quat.from_euler_zyx(0., np.pi/2+0.2, 0.)
        recording_camera = create_recording_camera(
            env.gym,
            env.envs[0],
            transform= transform,
        )
    img_idx=0
    for count in tqdm(range(total_runs)):

        camera_center=env.root_states[camera_follow_id, :3].cpu().numpy()
        camera_center[0]=2.5
        camera_center[1]=camera_center[1]+0.5
        camera_center[2]=0.32

        if MOVE_CAMERA:
            if CAMERA_FOLLOW:
                camera_position[:] = camera_center - camera_direction
            else:
                camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if RECORD_FRAMES:
            filename = os.path.join(
                os.path.abspath("logs/images/"),
                f"{img_idx:04d}.png",
            )
            print('filename:',filename)
            # env.gym.write_viewer_image_to_file(env.viewer, filename)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.write_camera_image_to_file(
                env.sim,
                env.envs[0],
                recording_camera,
                gymapi.IMAGE_COLOR,
                filename,
            )
            img_idx += 1

        obs = env.get_observations()
        positions = env.root_states[:,:3].cpu().numpy()
        distances = np.linalg.norm(positions[:, :2]- target_position, axis=1)
        within_target = distances < 0.2  

        directions = target_position - positions[:, :2]
        directions[:,1]=np.clip(directions[:,1], -3, 3)
        env._draw_guide_position(target_position)
        _, _, current_angles = get_euler_xyz(env.root_states[:, 3:7])
        current_angles = current_angles.cpu().numpy()
        
        cos_angles = np.cos(-current_angles)
        sin_angles = np.sin(-current_angles)
        local_directions_x = cos_angles * directions[:, 0] - sin_angles * directions[:, 1]
        local_directions_y = sin_angles * directions[:, 0] + cos_angles * directions[:, 1]
        directions = np.stack([local_directions_x, local_directions_y], axis=1)
        directions[:, 0] = np.clip(directions[:, 0], -4, 4) 
        directions[:, 1] = np.clip(directions[:, 1], -2, 2) 
        directions = torch.tensor(directions, dtype=torch.float32, device=env.device)

        if count<100:
            obs[:,9]=0
            obs[:,10]=0
            obs[:,48]=0.0
        else:
            obs[:,9]=directions[:,0]
            obs[:,10]=directions[:,1]*2
            obs[:,48]=3.0

        # stepping
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        success_count += (within_target * ~terminated).sum()
        finish_time[within_target * ~terminated] = count
        travel_distance[~terminated] = env.root_states[~terminated,1].cpu().numpy()
        
        deaths = dones.cpu().numpy() * ~terminated 
        if deaths.sum() > 0:
            print(f"Deaths occurred: {deaths.sum()}")
        
        terminated |= deaths
        terminated |= within_target
        print(f"Death number: {terminated.sum()}")
        print(f"Success_count: {success_count}")

    idx=finish_time==0
    finish_time[idx]=total_runs
    print(f"Finish_time: {finish_time.mean()/50}")

    success_rate = success_count / env.num_envs
    print(f"Success rate: {success_rate:.3f}")
    print(f"Travel distance: {travel_distance.mean()}")


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA =False
    CAMERA_FOLLOW = MOVE_CAMERA
    args = get_args([
        dict(name= "--slow", type= float, default= 0., help= "slow down the simulation by sleep secs (float) every frame"),
        dict(name= "--zero_act_until", type= int, default= 0., help= "zero action until this step"),
        dict(name= "--sample", action= "store_true", default= False, help= "sample actions from policy"),
        dict(name= "--plot_time", type= int, default= 100, help= "plot states after this time"),
        dict(name= "--no_throw", action= "store_true", default= False),
    ])
    track(args)