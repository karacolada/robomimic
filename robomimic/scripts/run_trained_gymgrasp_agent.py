"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    n_rollouts (int): number of rollouts to be run, must be either smaller or divisible by num_envs

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    seed (int): if provided, set seed for rollouts

    record (bool): if flag is provided, record rollouts into runs/recordings/{task_name}/

    n_envs (int): if provided, overwrite number of parallel environments

    resolution (str): (optional) desired resolution in format <width>x<height>, default is 128x128

Example usage:

    # Evaluate a policy with 50 rollouts in parallel of maximum horizon 400 and save the recordings in 900x900 videos.
    
    python run_trained_gymgrasp_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --n_envs 50 --horizon 400 --seed 0 \
        --record --resolution 
"""

import isaacgym

import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.train_utils import run_parallel_rollouts
from robomimic.envs.env_base import EnvType

def run_trained_agent(args):
    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    assert ckpt_dict["env_metadata"]["type"] == EnvType.GYMGRASP_TYPE

    env_args = ckpt_dict["env_metadata"]["env_kwargs"]
    if args.n_envs:
        env_args["task"]["env"]["numEnvs"] = args.n_envs
    elif args.n_rollouts < env_args["task"]["env"]["numEnvs"]:
        env_args["task"]["env"]["numEnvs"] = args.n_rollouts
    num_envs = env_args["task"]["env"]["numEnvs"]
    env_args["headless"] = True
    env_args["task"]["control"]["teleoperated"] = False

    if args.record:
        w, h = args.resolution.split('x')
        camera_0 = dict(type="rgb",
                        pos=[ -0.5, 0, 1.3 ],
                        lookat=[ 0,  0, 0.8 ],
                        horizontal_fov=70,
                        width=int(w),
                        height=int(h),
                    )
        cameras = dict(save_recordings=True,
                       convert_to_pointcloud=False,
                       convert_to_voxelgrid=False,
                       camera0=camera_0)
        env_args["task"]["cameras"] = cameras

    # read rollout settings
    assert args.n_rollouts % num_envs == 0
    rollout_num_episodes = int(args.n_rollouts / num_envs)
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=False, 
        render_offscreen=False, 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("============= Starting Rollouts =============")
    
    rollout_stats = []
    for i in range(rollout_num_episodes):
        rollout_info = run_parallel_rollouts(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=None,
            terminate_on_success=True
            )
        rollout_stats.append(rollout_info)
        print(f"Iteration {i} rollout stats:")
        print(rollout_info)
    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(avg_rollout_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--record",
        action='store_true',
        help="record rollouts in simulation",
    )

    parser.add_argument(
        "--n_envs",
        type=int,
        default=None,
        help="(optional) override number of parallel envs of rollout from the one in the checkpoint",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="128x128",
        help="(optional) desired resolution in format <width>x<height>, default is 128x128",
    )    

    args = parser.parse_args()
    run_trained_agent(args)

