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
    "name": "thesis-hyperparam-sweep-vanilla",
    "method": "random",
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
        "algo.actor.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_3_dim": {
            "values": [0, 256, 512]
        },
        "algo.critic.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_3_dim": {
            "values": [0, 256, 512]
        },
        "algo.optim_params.critic.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.critic.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "algo.optim_params.actor.learning_rate.decay_factor": {
            "values": [0.0, 0.1]
        },
        "algo.optim_params.actor.regularization.L2": {
            "values": [0.00, 0.01, 0.1]
        },  # L2 regularization strength, weight decay
        "algo.actor.max_gradient_norm": {
            "values": [None, 0.5]
        }, # L2 gradient clipping for actor, probably shouldn't use with weight decay i.e. L2 regularisation
        "algo.critic.use_huber": {
            "values": [True, False]
        },
        "algo.critic.max_gradient_norm": {
            "values": [None, 0.5]
        },
        "algo.critic.num_action_samples": {
            "values": [1, 2, 4, 10, 30]
        },
        "algo.critic.target_q_gap":{
            "values": [1.0, 5.0, 10.0]
        },
        #"train.hdf5_normalize_obs": {
        #    "values": [True, False]
        #},
        "train.seq_length": {
            "values": [1, 2, 4, 10, 30]
        },  # will be applied to n_step
        "train.batch_size": {
            "values": [100, 512, 1024]
        }
    }
}

sweep_config_rnn = {
    "name": "thesis-hyperparam-sweep-rnn",
    "method": "random",
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
        "algo.actor.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_3_dim": {
            "values": [0, 256, 512]
        },
        "algo.critic.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_3_dim": {
            "values": [0, 256, 512]
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
            "values": [None, 0.5]
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
            "values": [None, 0.5]
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
        #"train.hdf5_normalize_obs": {
        #    "values": [True, False]
        #},
        "train.batch_size": {
            "values": [100, 512, 1024]
        }
    }
}

sweep_config_ext = {
    "name": "thesis-hyperparam-sweep-ext",
    "method": "random",
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
        "algo.actor.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.actor.layer_3_dim": {
            "values": [0, 256, 512]
        },
        "algo.critic.layer_1_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_2_dim": {
            "values": [256, 512]
        },
        "algo.critic.layer_3_dim": {
            "values": [0, 256, 512]
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
            "values": [None, 0.5]
        }, # L2 gradient clipping for actor, probably shouldn't use with weight decay i.e. L2 regularisation
        "algo.critic.use_huber": {
            "values": [True, False]
        },
        "algo.critic.max_gradient_norm": {
            "values": [None, 0.5]
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
        #"train.hdf5_normalize_obs": {
        #    "values": [True, False]
        #},
        "train.batch_size": {
            "values": [100, 512, 1024]
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    if args.variant == "vanilla":
        sweep_id = wandb.sweep(sweep_config_vanilla, project=f"thesis-human-{args.task}-{args.variant}")
    elif args.variant == "rnn":
        sweep_id = wandb.sweep(sweep_config_rnn, project=f"thesis-human-{args.task}-{args.variant}")
    elif args.variant == "ext":
        sweep_id = wandb.sweep(sweep_config_ext, project=f"thesis-human-{args.task}-{args.variant}")
    else:
        print("Error: selected variant {}, need to choose from vanilla, rnn, ext.".format(args.variant))
        exit()
    print(sweep_id)
