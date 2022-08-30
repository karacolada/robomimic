from os import sync
from turtle import update
import wandb
import isaacgym
import json
import traceback

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train

base_config_path = "/home/karacol/git/learning-from-demonstrations/cfg/gen_configs/cql_lift_ph.json"

sweep_config = {
    "name": "thesis-hyperparam-sweep",
    "method": "random",
    "parameters": {
        "algo.optim_params.critic.learning_rate.initial": {
          "min": 0.0001,
          "max": 0.1
        },
        "algo.optim_params.critic.regularization.L2": {
          "values": [0.0, 0.01]
        },
        "algo.optim_params.actor.learning_rate.initial": {
          "min": 0.0001,
          "max": 0.1
        },
        "algo.optim_params.actor.regularization.L2": {
          "values": [0.0, 0.01]
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
        "algo.critic.layer_1_dim": {
          "values": [256, 512]
        },
        "algo.critic.layer_2_dim": {
          "values": [256, 512]
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
    l1 = wandb_config.pop("algo.actor.layer_1_dim")
    l2 = wandb_config.pop("algo.actor.layer_2_dim")
    wandb_config["algo.actor.layer_dims"] = [l1, l2]
    l1 = wandb_config.pop("algo.critic.layer_1_dim")
    l2 = wandb_config.pop("algo.critic.layer_2_dim")
    wandb_config["algo.critic.layer_dims"] = [l1, l2]
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
        #res_str = "finished run successfully!"
        #try:
        #    train(config, device=device)
        #except Exception as e:
        #    res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        #print(res_str)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=wrap_train)