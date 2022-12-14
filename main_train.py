import argparse
import os

from rllib.Model import Model

os.environ['WANDB_SILENT'] = "true"


def main():
    """
    trains model with specified params
    """
    # get command line args
    args = parse_args()
    env = args.env
    model_name = args.model_name
    num_epochs = args.num_epochs
    wandb_proj = args.wandb_project
    wandb_entity = args.wandb_entity
    val_check_interval = args.val_check_interval
    seed = args.seed

    # creates the model and sets up logging
    m = Model(seed=seed)
    m.create_model(env, model_name)
    m.create_wandb_logger(wandb_proj, wandb_entity)

    # trains the model
    m.train(num_epochs, val_check_interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="LunarLander-v2")
    parser.add_argument("-m", "--model_name", type=str, default="ppo_ex")
    # parser.add_argument("-nm", "--num_models", type=int, default=5)
    parser.add_argument("-ne", "--num_epochs", type=int, default=150)
    parser.add_argument("--val_check_interval", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default="rl_project")
    parser.add_argument("--wandb_entity", type=str, default="ece517")
    parser.add_argument("--seed", type=str, default="123")
    return parser.parse_args()


if __name__ == "__main__":
    main()
