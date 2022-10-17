import argparse
import wandb
import isaacgym
import json
import traceback
import time

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train

sweep_config_vanilla = {
    "name": "sweep-vanilla",
    "method": "grid",
    "program": "scan_hyperparam.py",
    "parameters": {
        "algo.optim_params.critic.learning_rate.initial": {
            "values": [1e-5, 3e-4]
        },
        "algo.optim_params.actor.learning_rate.initial": {
            "values": [1e-2, 3e-5, 3e-3]
        },
        "algo.actor.layer_dims":{
            "values": [(256, 256), (256, 512, 256)]
        },
        "algo.critic.layer_dims":{
            "values": [(256, 256), (512, 512)]
        },
        "algo.critic.target_q_gap":{
            "values": [1.0, 5.0]
        },
        #"train.hdf5_normalize_obs": {
        #    "values": [True, False],
        #    #"probabilities": [0.2, 0.8]  # low probability bc. we need to disable validation
        #}
    }
}

sweep_config_rnn = {
    "name": "sweep-rnn",
    "method": "random",
    "program": "scan_hyperparam.py",    
    "parameters": {
        "algo.optim_params.critic.learning_rate.initial": {
            "values": [1e-5, 1e-4, 1e-3, 1e-2, 3e-5, 3e-4, 3e-3, 3e-2]
        },
        "algo.optim_params.actor.learning_rate.initial": {
            "values": [1e-5, 1e-4, 1e-3, 1e-2, 3e-5, 3e-4, 3e-3, 3e-2]
        },
        "algo.target_tau": {
            "values": [0.005, 0.0005]
        },
        "algo.actor.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.critic.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.optim_params.critic.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.critic.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },
        "algo.optim_params.actor.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.actor.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "algo.actor.max_gradient_norm": {
            "values": [-1, 0.5]
        },
        "algo.actor.net.rnn.enabled": {
            "values": [True, False]
        },
        "algo.actor.net.rnn.hidden_dim": {
            "values": [256, 512]
        },
        "algo.actor.net.rnn.num_layers": {
            "values": [1, 2, 3]
        },
        "algo.actor.net.rnn.remove_mlp": {
            "values": [True, False]
        },  # use to ignore given MLP layers
        "algo.critic.use_huber": {
            "values": [True, False]
        },
        "algo.critic.max_gradient_norm": {
            "values": [-1, 0.5]
        },
        "algo.critic.num_action_samples": {
            "values": [1, 2, 4, 10, 30]
        },
        "algo.critic.target_q_gap":{
            "values": [1.0, 5.0, 10.0]
        },
        "algo.critic.rnn.remove_mlp": {
            "values": [True, False]
        },  # use to ignore given MLP layers
        "algo.critic.rnn.shared": {
            "values": [True, False]
        },
        "algo.critic.rnn.hidden_dim": {
            "values": [256, 512]
        },
        "algo.critic.rnn.num_layers": {
            "values": [1, 2, 3]
        },
        "train.seq_length": {
            "values": [2, 4, 10, 30]
        },
        "train.hdf5_normalize_obs": {
            "values": [True, False],
            "probabilities": [0.2, 0.8]  # low probability bc. we need to disable validation
        },
        "train.batch_size": {
            "values": [100, 512, 1024]
        }
    }
}

sweep_config_ext = {
    "name": "sweep-ext",
    "method": "random",
    "program": "scan_hyperparam.py",
    "parameters": {
        "algo.optim_params.critic.learning_rate.initial": {
            "values": [1e-5, 1e-4, 1e-3, 1e-2, 3e-5, 3e-4, 3e-3, 3e-2]
        },
        "algo.optim_params.actor.learning_rate.initial": {
            "values": [1e-5, 1e-4, 1e-3, 1e-2, 3e-5, 3e-4, 3e-3, 3e-2]
        },
        "algo.target_tau": {
            "values": [0.005, 0.0005]
        },
        "algo.actor.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.critic.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.optim_params.critic.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.critic.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },
        "algo.optim_params.actor.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.actor.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "algo.actor.max_gradient_norm": {
            "values": [-1, 0.5]
        }, # L2 gradient clipping for actor, probably shouldn't use with weight decay i.e. L2 regularisation
        "algo.critic.use_huber": {
            "values": [True, False]
        },
        "algo.critic.max_gradient_norm": {
            "values": [-1, 0.5]
        },
        "algo.critic.num_action_samples": {
            "values": [1, 2, 4, 10, 30]
        },
        "algo.critic.target_q_gap":{
            "values": [1.0, 5.0, 10.0]
        },
        "train.seq_length": {
            "values": [2, 4, 10, 30]
        },
        "train.hdf5_normalize_obs": {
            "values": [True, False],
            "probabilities": [0.2, 0.8]  # low probability bc. we need to disable validation
        },
        "train.batch_size": {
            "values": [100, 512, 1024]
        }
    }
}

sweep_config_robomimic_rnn = {
    "name": "sweep-rnn",
    "method": "random",
    "program": "scan_hyperparam.py",    
    "parameters": {
        "algo.optim_params.critic.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },
        "algo.optim_params.actor.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "algo.actor.net.rnn.enabled": {
            "values": [True, False]
        },
        "algo.actor.net.rnn.hidden_dim": {
            "values": [256, 512]
        },
        "algo.actor.net.rnn.num_layers": {
            "values": [1, 2, 3]
        },
        "algo.actor.net.rnn.remove_mlp": {
            "values": [True, False]
        },  # use to ignore given MLP layers
        "algo.critic.rnn.remove_mlp": {
            "values": [True, False]
        },  # use to ignore given MLP layers
        "algo.critic.rnn.shared": {
            "values": [True, False]
        },
        "algo.critic.rnn.hidden_dim": {
            "values": [256, 512]
        },
        "algo.critic.rnn.num_layers": {
            "values": [1, 2, 3]
        },
        "train.seq_length": {
            "values": [2, 4, 10, 30]
        }
    }
}

sweep_config_robomimic_ext = {
    "name": "sweep-ext",
    "method": "random",
    "program": "scan_hyperparam.py",
    "parameters": {
        "algo.actor.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.critic.layer_dims":{
            "values": [(256, 256), (256, 512, 256), (512, 512)]
        },
        "algo.optim_params.critic.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },
        "algo.optim_params.actor.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "train.seq_length": {
            "values": [2, 4, 10, 30]
        }
    }
}

def register_sweep(suite, variant, task):
    if "gym-grasp" in suite:
        if suite == "gym-grasp-human":
            project = f"thesis-human-{task}"
        elif suite == "gym-grasp-mg":
            project = f"thesis-mg-{task}"
        if variant == "vanilla":
            sweep_cfg = sweep_config_vanilla
        elif variant == "rnn":
            sweep_cfg = sweep_config_rnn
        elif variant == "ext":
            sweep_cfg = sweep_config_ext
        else:
            print("Error: selected variant {} is not available, need to choose from vanilla, rnn, ext.".format(variant))
            exit()
        if task == "OpenDrawer":  # not enough trajectories in dataset for batch size 1024
            sweep_cfg["parameters"]["train.batch_size"]["values"].remove(1024)
    elif suite == "robomimic":
        project = f"thesis-ph-{task}"
        if variant == "rnn":
            sweep_cfg = sweep_config_robomimic_rnn
        elif variant == "ext":
            sweep_cfg = sweep_config_robomimic_ext
        elif variant == "vanilla":
            print("Error: vanilla variant is not available for robomimic sweep. See Mandlekar et al. paper, they already did that.")
            exit()
        else:
            print("Error: selected variant {} is not available, need to choose from rnn, ext.".format(variant))
            exit()
    sweep_id = wandb.sweep(sweep_cfg, project=project)
    print(sweep_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="one of gym-grasp-human, gym-grasp-mg, robomimic",
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="one of vanilla, rnn, ext",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="one of OpenDrawer, OpenDoor, PourCup",
    )

    args = parser.parse_args()
    register_sweep(args.suite, args.variant, args.task)
