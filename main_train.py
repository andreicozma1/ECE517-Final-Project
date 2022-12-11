import argparse

from rllib.Utils import Model


def main():
    args = parse_args()
    env = args.env
    model_name = args.model_name
    num_epochs = args.num_epochs
    wandb_proj = args.wandb_project
    val_check_interval = args.val_check_interval

    m = Model()
    m.create_model(env, model_name)
    m.create_wandb_logger(wandb_proj)
    m.train(num_epochs, val_check_interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="LunarLander-v2")
    parser.add_argument("-m", "--model_name", type=str, default="ppo_ex")
    parser.add_argument("-nm", "--num_models", type=int, default=5)
    parser.add_argument("-ne", "--num_epochs", type=int, default=150)
    parser.add_argument("--val_check_interval", type=int, default=25)
    parser.add_argument("--wandb_project", type=str, default="rl_project")
    return parser.parse_args()


if __name__ == "__main__":
    main()
