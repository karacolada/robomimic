"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    n_rollouts (int): number of rollouts to be run

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    seed (int): if provided, set seed for rollouts

    record (bool): if flag is provided, record rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the recordings.
    
    python run_trained_gymgrasp_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --record
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

def run_trained_agent(args):
    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    num_envs = ckpt_dict["env_metadata"]["env_kwargs"]["task"]["env"]["numEnvs"]
    ckpt_dict["env_metadata"]["env_kwargs"]["headless"] = True

    if args.record:
        camera_0 = dict(type="rgb",
                        pos=[ -0.5, 0, 1.3 ],
                        lookat=[ 0,  0, 0.8 ],
                        horizontal_fov=70,
                        width=900,
                        height=900,
                    )
        cameras = dict(save_recordings=True,
                       convert_to_pointcloud=False,
                       convert_to_voxelgrid=False,
                       camera0=camera_0)
        ckpt_dict["env_metadata"]["env_kwargs"]["task"]["cameras"] = cameras

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

    args = parser.parse_args()
    run_trained_agent(args)

