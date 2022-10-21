import wandb
import isaacgym
import json
import traceback

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train

base_config_path = ""
suite = "robomimic"  # one of robomimic, gym-grasp
variant = "vanilla"  # one of vanilla, rnn, ext

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

def apply_wandb_conf(config, wandb_config, run_id):
    if suite == "gym-grasp":
        # clear up max_gradient_clipping
        #for m in ["actor", "critic"]:
        #    if wandb_config[f"algo.{m}.max_gradient_norm"] == -1:
        #        wandb_config[f"algo.{m}.max_gradient_norm"] = None
        # lr decay scheduler
        try:
            for m in ["actor", "critic"]:
                decay_factor = wandb_config[f"algo.optim_params.{m}.learning_rate.decay_factor"]
                if decay_factor > 0:
                    wandb_config[f"algo.optim_params.{m}.learning_rate.epoch_schedule"] = [200, 400, 600, 800]
        except KeyError:
            pass
        # observation normalisation does not work with validation
        try:
            if wandb_config["train.hdf5_normalize_obs"]:
                wandb_config["experiment.validate"] = False
        except KeyError:
            pass
    # RNN drop MLP
    if variant == "rnn":
        # remove MLP
        actor_remove_mlp = wandb_config.pop("algo.actor.net.rnn.remove_mlp")
        critic_remove_mlp = wandb_config.pop("algo.critic.rnn.remove_mlp")
        if critic_remove_mlp:
            wandb_config["algo.critic.layer_dims"] = ()
        if actor_remove_mlp and wandb_config["algo.actor.net.rnn.enabled"]:
            wandb_config["algo.actor.layer_dims"] = ()
        # RNN layer config
        actor_rnn_layers = wandb_config.pop("algo.actor.net.rnn.layer_dims")
        if actor_rnn_layers == (0):
            wandb_config["algo.actor.net.rnn.enabled"] = False
        else:
            wandb_config["algo.actor.net.rnn.enabled"] = False
            wandb_config["algo.actor.net.rnn.hidden_dim"] = actor_rnn_layers[0]
            wandb_config["algo.actor.net.rnn.num_layers"] = len(actor_rnn_layers)
        critic_rnn_layers = wandb_config.pop("algo.critic.rnn.layer_dims")
        wandb_config["algo.critic.rnn.hidden_dim"] = critic_rnn_layers[0]
        wandb_config["algo.critic.rnn.num_layers"] = len(critic_rnn_layers)
    # sequence length
    if variant == "rnn":
        wandb_config["algo.actor.net.rnn.horizon"] = wandb_config["train.seq_length"]
        wandb_config["algo.critic.rnn.horizon"] = wandb_config["train.seq_length"]
    elif variant == "ext":
        wandb_config["algo.ext.history_length"] = wandb_config["train.seq_length"]
    elif variant == "vanilla":
        try:
            wandb_config["algo.n_step"] = wandb_config["train.seq_length"]
        except KeyError:
            pass
    # adjust log directory
    wandb_config["experiment.name"] = config["experiment"]["name"] + "_" + run_id
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
        config = apply_wandb_conf(config, wandb.config._as_dict(), run.id)

        # lock config to prevent further modifications and ensure missing keys raise errors
        config.lock()

        # catch error during training and print it
        res_str = "finished run successfully!"
        try:
            train(config, device=device)
        except Exception as e:
            res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)

if __name__ == "__main__":
    wrap_train()
