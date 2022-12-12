import collections

import argparse

import numpy as np
from tqdm import tqdm

from rllib.Model import Model


def main():
    args = parse_args()
    running_rew_len = args.running_rew_len
    num_episodes = args.num_episodes
    checkpoint_path = args.file_path
    seed = args.seed

    m = Model(seed=seed)

    running_reward: collections.deque = collections.deque(maxlen=running_rew_len)
    tq_episode_iter = tqdm(range(num_episodes), leave=False, desc="Episode")
    for _ in tq_episode_iter:
        m.load_model(checkpoint_path)
        metrics = m.eval()
        running_reward.append(metrics['total_reward'])
        metrics |= {
                "running_reward": np.mean(running_reward),
        }
        tq_episode_iter.set_postfix(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("-f", "--file_path", type=str, required=True)
    parser.add_argument("-ne", "--num_episodes", type=int, default=5)
    parser.add_argument("--running_rew_len", type=int, default=50)
    parser.add_argument("--seed", type=str, default="123")
    return parser.parse_args()


if __name__ == "__main__":
    main()
