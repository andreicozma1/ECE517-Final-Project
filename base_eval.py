import collections
import hashlib
import os

import argparse
# import gym
import sys
import time

import numpy as np
import torch
# from pl_bolts.models.rl.ppo_model import PPO
import tqdm
from keras.callbacks import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from rllib.advantage_actor_critic_model import AdvantageActorCritic
from rllib.custompp import CustomPPO
from rllib.ppo_model import PPO


# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning import Trainer


def eval_model(i, args, model_params={}):
    checkpoint_path = args.file_path
    model_path = checkpoint_path.split('/')
    model_name, env = model_path[-3], model_path[-2]
    print("=" * 80)
    print(model_name)
    print(f" - {model_name}")
    print(f" - {env}")

    model = get_model(env, model_name, model_params)
    model.load_state_dict(torch.load(checkpoint_path))

    # setup environemnt
    done = False
    state = torch.FloatTensor(model.env.reset())
    total_rewards, total_steps = 0, 0
    pred_value = []
    hist_rewards = []
    while not done:
        state = state.to(device=model.device)
        next_state, value, reward, done = None, None, None, None
        # api for models are a bit different for getting state and value
        if model_name == "ppo":
            with torch.no_grad():
                pi, action, value = model(state)
            next_state, reward, done, _ = model.env.step(action.cpu().numpy())

        elif model_name == "a2c":
            with torch.no_grad():
                action = model.agent(state, model.device)[0]
                _, value = model.net(state.reshape((1, -1)))
            next_state, reward, done, _ = model.env.step(action.cpu().numpy())

        hist_rewards.append(reward)
        pred_value.append(value)

        # Render the env
        model.env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)
        state = torch.FloatTensor(next_state)

    actual_return = discount_rewards(hist_rewards, model.gamma)

    # close the render
    model.env.close()
    return {
            "total_reward" : np.sum(hist_rewards),
            "total_steps"  : total_steps,
            'actual_return': actual_return,
            'pred_value'   : pred_value
    }


def get_model(env, model_name, model_params):
    # setup model
    models = {
            "ppo"      : PPO(env, *model_params),
            "a2c"      : AdvantageActorCritic(env, *model_params),
            "CustomPPO": CustomPPO(env, *model_params),
    }
    if model_name not in models:
        print(f"ERROR: Model {model_name} not supported")
        print("Available models:")
        for model in models:
            print(f" - {model}")
        sys.exit(1)
    model = models[model_name]
    return model


def discount_rewards(rewards, discount: float):
    """Calculate the discounted rewards of all rewards in list.

    Args:
        rewards: list of rewards/advantages
        discount: discount factor

    Returns:
        list of discounted rewards/advantages
    """

    cumul_reward = []
    sum_r = 0.0

    for r in reversed(rewards):
        sum_r = (sum_r * discount) + r
        cumul_reward.append(sum_r)

    return list(reversed(cumul_reward))


def main():
    args = parse_args()
    running_rew_len = args.running_rew_len
    num_episodes = args.num_episodes

    running_reward: collections.deque = collections.deque(maxlen=running_rew_len)
    tq_episodes = tqdm.trange(num_episodes, leave=False)
    for i in tq_episodes:
        metrics = eval_model(i, args)
        running_reward.append(metrics['total_reward'])
        tq_episodes.set_postfix(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("-f", "--file_path", type=str, required=True)
    parser.add_argument("-ne", "--num_episodes", type=int, default=1)
    parser.add_argument("--running_rew_len", type=int, default=50)
    # Logging and metrics
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--wandb_project", type=str, default="rl_project")
    return parser.parse_args()


if __name__ == "__main__":
    main()
