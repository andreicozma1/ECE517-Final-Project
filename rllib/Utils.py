import hashlib
import sys

import torch
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from rllib.PPO1 import PPO1
from rllib.examples.A2CExample import AdvantageActorCritic
from rllib.examples.PPOExample import PPO


def get_model(env, model_name, model_params):
    # setup model
    models = {
            "ppo_ex": PPO(env, *model_params),
            "a2c_ex": AdvantageActorCritic(env, *model_params),
            "ppo_1" : PPO1(env, *model_params),
    }
    if model_name not in models:
        print(f"ERROR: Model {model_name} not supported")
        print("Available models:")
        for model in models:
            print(f" - {model}")
        sys.exit(1)
    model = models[model_name]
    return model


def load_model(checkpoint_path, model_params):
    model_path = checkpoint_path.split('/')
    model_name, env = model_path[-3], model_path[-2]
    model = get_model(env, model_name, model_params)
    model.load_state_dict(torch.load(checkpoint_path))
    return model, model_name


def get_model_hash(model):
    return str(hashlib.md5(str(model).encode('utf-8')).hexdigest())


def get_logger(project, log_dir, model_hash, model_name):
    project = project.strip()
    if project == "":
        return CSVLogger(save_dir=log_dir, name=model_hash)

    return WandbLogger(project=project, name=model_hash,
                       group=model_name, tags=["train"])


def discount_rewards(rewards: list, gamma: float) -> list:
    cumul_reward = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r = (sum_r * gamma) + r
        cumul_reward.append(sum_r)
    return list(reversed(cumul_reward))
