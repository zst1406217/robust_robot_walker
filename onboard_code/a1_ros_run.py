import os
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Tuple

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
import ros_numpy

from geometry_msgs.msg import Twist
from a1_real import UnitreeA1Real, resize2d
from rsl_rl import modules
from rsl_rl.utils.utils import get_obs_slice

from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
import sys
import pyrealsense2 as rs2
from std_msgs.msg import Float32MultiArray

@torch.no_grad()
def handle_forward_depth(ros_msg, model, publisher, output_resolution, device):
    """ The callback function to handle the forward depth and send the embedding through ROS topic """
    buf = ros_numpy.numpify(ros_msg).astype(np.float32)
    forward_depth_buf = resize2d(
        torch.from_numpy(buf).unsqueeze(0).unsqueeze(0).to(device),
        output_resolution,
    )
    embedding = model(forward_depth_buf)
    ros_data = embedding.reshape(-1).cpu().numpy().astype(np.float32)
    publisher.publish(Float32MultiArray(data= ros_data.tolist()))

class ImageListener:
    def __init__(self, topic):
        self.topic = topic
        self.sub = rospy.Subscriber(topic, msg_Image, self.imageDepthCallback)
        self.sub_request = rospy.Subscriber('/axis_pub', Float32MultiArray, self.request_callback)
        self.sub_info = rospy.Subscriber('/camera_down/aligned_depth_to_color/camera_info', CameraInfo, self.imageDepthInfoCallback)
        self.pub_request = rospy.Publisher('/depth_axis', Float32MultiArray, queue_size=1)
        self.msg=Float32MultiArray()
        self.intrinsics = None
        self.height = None
        self.width = None

    def imageDepthCallback(self, data):
        self.cv_image = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height,data.width)
        if self.height==None:
            print("camera_OK")
        self.height=data.height
        self.width=data.width

    def request_callback(self, data):
        array=data.data
        pix_x=round(self.width*(array[0]+array[2])/2)
        # pix_y=round(self.height*(array[1]+array[3])/2)
        pix_y=round(self.height*array[3])
        pix=[pix_x,pix_y]
        print("OK1")
        if self.intrinsics:
            print("OK2")
            depth = self.cv_image[pix[1], pix[0]]
            result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
            result=[result[0]/1000., result[1]/1000., result[2]/1000.]
            self.msg.data=result
            self.pub_request.publish(self.msg)

    def imageDepthInfoCallback(self, cameraInfo):
        # import pdb; pdb.set_trace()
        if self.intrinsics:
            return
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]

class StandOnlyModel(torch.nn.Module):
    def __init__(self, action_scale, dof_pos_scale, tolerance= 0.1, delta= 0.1):
        rospy.loginfo("Using stand only model, please make sure the proprioception is 48 dim.")
        rospy.loginfo("Using stand only model, -36 to -24 must be joint position.")
        super().__init__()
        if isinstance(action_scale, (tuple, list)):
            self.register_buffer("action_scale", torch.tensor(action_scale))
        else:
            self.action_scale = action_scale
        if isinstance(dof_pos_scale, (tuple, list)):
            self.register_buffer("dof_pos_scale", torch.tensor(dof_pos_scale))
        else:
            self.dof_pos_scale = dof_pos_scale
        self.tolerance = tolerance
        self.delta = delta

    def forward(self, obs):
        joint_positions = obs[..., 12:24] / self.dof_pos_scale
        diff_large_mask = torch.abs(joint_positions) > self.tolerance
        target_positions = torch.zeros_like(joint_positions)
        target_positions[diff_large_mask] = joint_positions[diff_large_mask] - self.delta * torch.sign(joint_positions[diff_large_mask])
        return torch.clip(
            target_positions / self.action_scale,
            -1.0, 1.0,
        )
    
    def reset(self, *args, **kwargs):
        pass

def load_walk_policy(env, model_dir):
    """ Load the walk policy from the model directory """
    if model_dir == None:
        model = StandOnlyModel(
            action_scale= env.action_scale,
            dof_pos_scale= env.obs_scales["dof_pos"],
        )
        policy = torch.jit.script(model)

    else:
        with open(osp.join(model_dir, "config.json"), "r") as f:
            config_dict = json.load(f, object_pairs_hook= OrderedDict)
        obs_components = config_dict["env"]["obs_components"]
        privileged_obs_components = config_dict["env"].get("privileged_obs_components", obs_components)
        estimated_components = config_dict["env"]["estimated_obs_components"]
        config_dict["policy"]["num_scan"]=187
        config_dict["policy"]["num_estimated"]=env.get_num_estimated_from_components(estimated_components)
        num_dim=0
        if "foot" in config_dict["asset"]["collision_body_names"]:
            num_dim+=4
        if "thigh" in config_dict["asset"]["collision_body_names"]:
            num_dim+=4
        if "calf" in config_dict["asset"]["collision_body_names"]:
            num_dim+=4
        if "hip" in config_dict["asset"]["collision_body_names"]:
            num_dim+=4
        if "base" in config_dict["asset"]["collision_body_names"]:
            num_dim+=1
        config_dict["policy"]["num_stumble"]=num_dim
        model = getattr(modules, config_dict["runner"]["policy_class_name"])(
            num_actor_obs= env.get_num_obs_from_components(obs_components),
            num_critic_obs= env.get_num_obs_from_components(privileged_obs_components),
            num_actions= 12,
            **config_dict["policy"],
        )
        model_names = [i for i in os.listdir(model_dir) if i.startswith("model_")]
        model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
        state_dict = torch.load(osp.join(model_dir, model_names[-1]), map_location= "cpu")
        model.load_state_dict(state_dict["model_state_dict"],strict=False)
        # model.load_state_dict(state_dict["model_state_dict_student"])
        estimator_cfg=config_dict["estimator"]
        estimator = getattr(modules, "Estimator")(
            input_dim=estimator_cfg["input_dim"],
            # output_dim=env.get_num_estimated_from_components(estimated_components), 
            output_dim=model.estimator_output_dim, 
            rnn_type=estimator_cfg["rnn_type"],
            rnn_hidden_size=estimator_cfg["rnn_hidden_size"],
            latent_encoder_hidden_dims=estimator_cfg["latent_encoder_hidden_dims"]
            )
        num_props=estimator_cfg["input_dim"]
        print("num_props:", num_props)
        print(estimator)
        estimator.load_state_dict(state_dict["estimator_state_dict"])
        model_action_scale = torch.tensor(config_dict["control"]["action_scale"]) if isinstance(config_dict["control"]["action_scale"], (tuple, list)) else torch.tensor([config_dict["control"]["action_scale"]])[0]
        if not (torch.is_tensor(model_action_scale) and (model_action_scale == env.action_scale).all()):
            action_rescale_ratio = model_action_scale / env.action_scale
            print("walk_policy action scaling:", action_rescale_ratio.tolist())
        else:
            action_rescale_ratio = 1.0
        
        est_memory=estimator.memory_a
        est_mlp=estimator.latent_encoder
        est_output=estimator.latent_output
        actor_memory=model.memory_a
        actor_mlp=model.actor

        @torch.jit.script
        def policy_run(obs):
            obs_prop = obs[:, 3:49]
            h=est_memory(obs_prop)
            predicted_latent = est_mlp(h)
            predicted_latent = est_output(predicted_latent).squeeze(0)
            obs_vel = predicted_latent[:, :3]
            obs_others = predicted_latent[:, 3:]
            backbone_input = torch.cat([obs_vel, obs_prop, obs_others], dim=1)
            tmp = actor_memory(backbone_input)
            actions = actor_mlp(tmp.squeeze(0))
            return actions
        
        if (torch.is_tensor(action_rescale_ratio) and (action_rescale_ratio == 1.).all()) \
            or (not torch.is_tensor(action_rescale_ratio) and action_rescale_ratio == 1.):
            policy = policy_run
        else:
            policy = lambda x: policy_run(x) * action_rescale_ratio
    
    return policy, model

class SkilledA1Real(UnitreeA1Real):
    """ Some additional methods to help the execution of skill policy """
    def __init__(self, *args,
            skill_mode_threhold= 0.1,
            skill_vel_range= [0.0, 1.0],
            max_vel=0.5,
            **kwargs,
        ):
        self.skill_mode_threhold = skill_mode_threhold
        self.skill_vel_range = skill_vel_range
        self.max_vel=max_vel
        super().__init__(*args, **kwargs)

    def is_skill_mode(self):
        return False

    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg
        return super().update_low_state(ros_msg,self.max_vel)

def main(args):
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("a1_legged_gym_" + "upboard", log_level= log_level)
    topic = '/camera_down/aligned_depth_to_color/image_raw'
    listener = ImageListener(topic)
    terrain_pub=rospy.Publisher('terrain_class', Int8, queue_size=1)
    terrain_data=Int8()

    with open(osp.join(args.walkdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # in sec
    # config_dict["control"]["stiffness"]["joint"] -= 2.5 # kp

    model_device = torch.device("cpu")

    unitree_real_env = SkilledA1Real(
        robot_namespace= args.namespace,
        cfg= config_dict,
        move_by_wireless_remote=True,
        skill_vel_range= config_dict["commands"]["ranges"]["lin_vel_x"],
        model_device= model_device,
        max_vel = args.max_vel
    )

    stand_model = StandOnlyModel(
            action_scale= unitree_real_env.action_scale,
            dof_pos_scale= unitree_real_env.obs_scales["dof_pos"],
        )
    stand_policy = torch.jit.script(stand_model)

    rospy.loginfo("duration: {}, motor Kp: {}, motor Kd: {}".format(
        duration,
        config_dict["control"]["stiffness"]["joint"],
        config_dict["control"]["damping"]["joint"],
    ))
    rospy.loginfo("[Env] torque limit: {:.1f}".format(unitree_real_env.torque_limits.mean().item()))
    rospy.loginfo("[Env] action scale: {:.1f}".format(unitree_real_env.action_scale))
    rospy.loginfo("[Env] motor strength: {}".format(unitree_real_env.motor_strength))
    
    # extract and build the torch ScriptFunction
    walk_policy, walk_model = load_walk_policy(unitree_real_env, args.walkdir)
    walk_model.reset()

    unitree_real_env.start_ros()
    unitree_real_env.wait_untill_ros_working()
    rate = rospy.Rate(1 / duration)
    with torch.no_grad():
        stand_flag=0
        while not rospy.is_shutdown():
            walk_obs = unitree_real_env._get_whole_obs()
            actions = walk_policy(walk_obs)
            if stand_flag==0:
                actions = stand_policy(walk_obs)
            if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.R1 and stand_flag==0:
                stand_flag=1
                rospy.loginfo("Robot standing up procedure finished!")
            unitree_real_env.send_action(actions)
            rate.sleep()
            if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.down:
                rospy.loginfo_throttle(0.1, "model reset")
                walk_model.reset()
            if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.L2 or unitree_real_env.low_state_buffer.wirelessRemote.btn.components.R2:
                unitree_real_env.publish_legs_cmd(unitree_real_env.default_dof_pos.unsqueeze(0), kp= 2, kd= 0.5)
                rospy.signal_shutdown("Controller send stop signal, exiting")

if __name__ == "__main__":
    """ The script to run the A1 script in ROS.
    It's designed as a main function and not designed to be a scalable code.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
        type= str,
        default= "/a112138",                    
    )
    parser.add_argument("--walkdir",
        type= str,
        help= "The log directory of the walking model.",
        default= None,
    )
    parser.add_argument("--max_vel",
        type= float,
        help= "Max velocity of the robot.",
        default= 0.8,
    )
    parser.add_argument("--debug",
        action= "store_true",
    )

    args = parser.parse_args()
    main(args)
