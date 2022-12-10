import collections

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from rllib.Utils import discount_rewards, load_model


def eval_model(episode_num, args, model_params={}):
    checkpoint_path = args.file_path
    model, model_name = load_model(checkpoint_path, model_params)

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


def main():
    args = parse_args()
    running_rew_len = args.running_rew_len
    num_episodes = args.num_episodes

    running_reward: collections.deque = collections.deque(maxlen=running_rew_len)
    tq_episode_iter = tqdm(range(num_episodes), leave=False, description="Episode")
    for episode_number in tq_episode_iter:
        metrics = eval_model(episode_number, args)
        running_reward.append(metrics['total_reward'])
        metrics |= {
                "running_reward": np.mean(running_reward),
        }
        tq_episode_iter.set_postfix(metrics)


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
