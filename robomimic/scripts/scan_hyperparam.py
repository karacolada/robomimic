import wandb
import isaacgym
import json
import traceback
import time

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train

base_config_path = "/home/user/moraw/learning-from-demonstrations/cfg/cql_eval/sweep_task_OpenDoor.json"
variant = "vanilla"  # one of vanilla, rnn, ext
task = "OpenDoor"

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

def recursive_init(config, key, value):
    if len(key) == 1:
        config[key[0]] = value
    else:
        if key[0] in config.keys():
            recursive_init(config[key[0]], key[1:], value)
        else:
            nested = {}
            recursive_init(nested, key[1:], value)
            config[key[0]] = nested

def nested_wandb(config):
    new_wandb = {}
    for k, v in config.items():
        if k != "_wandb":
            nested_k = k.split(".")
            recursive_init(new_wandb, nested_k, v)
    return new_wandb

def apply_wandb_conf(config, wandb_config):
    # adjust layer dims
    for m in ["actor", "critic"]:
        layer_dims = [wandb_config.pop(f"algo.{m}.layer_{i}_dim") for i in [1, 2, 3]]
        if layer_dims[2] == 0: # no third layer
            layer_dims = layer_dims[:2]
        if variant == "rnn":
            if m == "actor":
                remove_mlp = wandb_config.pop("algo.actor.net.rnn.remove_mlp")
            elif m == "critic":
                remove_mlp = wandb_config.pop("algo.critic.rnn.remove_mlp")
            if remove_mlp:
                layer_dims = []
        wandb_config[f"algo.{m}.layer_dims"] = layer_dims
    # lr decay scheduler
    for m in ["actor", "critic"]:
        decay_factor = wandb_config[f"algo.optim_params.{m}.learning_rate.decay_factor"]
        if decay_factor > 0:
            wandb_config[f"algo.optim_params.{m}.learning_rate.epoch_schedule"] = [200, 400, 600, 800]
    # sequence length
    if variant == "rnn":
        wandb_config["algo.actor.net.rnn.horizon"] = wandb_config["train.seq_length"]
        wandb_config["algo.critic.rnn.horizon"] = wandb_config["train.seq_length"]
    elif variant == "ext":
        wandb_config["algo.ext.history_length"] = wandb_config["train.seq_length"]
    elif variant == "vanilla":
        wandb_config["algo.n_step"] = wandb_config["train.seq_length"]
    # adjust outdir
    wandb_config["train.output_dir"] = config["train"]["output_dir"] + "-" + str(int(time.time()))
    # adjust config
    wandb_config = nested_wandb(wandb_config)
    with config.values_unlocked():
        config.update(wandb_config)
    return config

def wrap_train():
    with wandb.init(sync_tensorboard=True) as run:
        ext_cfg = json.load(open(base_config_path, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)

        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda, id=config.train.cuda_id)

        # wandb config stuff
        config = apply_wandb_conf(config, wandb.config._as_dict())

        # lock config to prevent further modifications and ensure missing keys raise errors
        config.lock()

        print(config)

        # catch error during training and print it
        res_str = "finished run successfully!"
        try:
            train(config, device=device)
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)

if __name__ == "__main__":
    if variant == "vanilla":
        sweep_id = wandb.sweep(sweep_config_vanilla, project=f"thesis-cql-{variant}-{task}")
    elif variant == "rnn":
        sweep_id = wandb.sweep(sweep_config_rnn, project=f"thesis-cql-{variant}-{task}")
    elif variant == "ext":
        sweep_id = wandb.sweep(sweep_config_ext, project=f"thesis-cql-{variant}-{task}")
    else:
        print("Error: selected variant {}, need to choose from vanilla, rnn, ext.".format(variant))
        exit()
    wandb.agent(sweep_id, function=wrap_train, count=50)