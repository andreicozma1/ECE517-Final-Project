import argparse
import hashlib
import os

# from pl_bolts.models.rl import AdvantageActorCritic
import sys

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# from pl_bolts.models.rl.ppo_model import PPO
from rllib.A2CExample import AdvantageActorCritic
from rllib.PPO1 import PPO1
from rllib.PPOExample import PPO


def train(args, model_params={}):
    env = args.env
    model_name = args.model_name
    num_epochs = args.num_epochs
    checkpoint_dir = args.checkpoint_dir
    wandb_proj = args.wandb_project
    log_dir = args.log_dir

    # setup model
    models = {
            "ppo"      : PPO(args.env, *model_params),
            "a2c"      : AdvantageActorCritic(args.env, *model_params),
            "CustomPPO": PPO1(args.env, *model_params),
    }

    if model_name not in models:
        print(f"ERROR: Model {model_name} not supported")
        print("Available models:")
        for model in models:
            print(f" - {model}")
        sys.exit(1)

    model = models[model_name]
    model_hash = get_model_hash(model)
    logger = get_logger(wandb_proj, log_dir, model_hash, model_name)
    # callbacks = []
    # if args.checkpoint_dir != "":
    #     checkpoint_callback = ModelCheckpoint(
    #             dirpath=args.checkpoint_dir,
    #             every_n_epochs=args.checkpoint_freq,
    #             verbose=True
    #         )
    #     callbacks.append(checkpoint_callback)

    # setup trainer
    trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=num_epochs,
            val_check_interval=args.val_check_interval,
            logger=logger,
            enable_checkpointing=True,
            auto_lr_find=True,
            auto_scale_batch_size=True,
            enable_model_summary=True,
            precision=16,
            # callbacks=callbacks
    )
    # train
    trainer.fit(model)
    model_save_dir = os.path.join(checkpoint_dir, model_name, env)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, f"{model_hash}.pt")
    print(save_path)
    torch.save(model.state_dict(), save_path)


def get_logger(project, log_dir, model_hash, model_name):
    project = project.strip()
    if project == "":
        return CSVLogger(save_dir=log_dir, name=model_hash)

    return WandbLogger(project=project, name=model_hash,
                       group=model_name, tags=["train"])


def get_model_hash(model):
    m_hash = hashlib.md5(str(model).encode('utf-8')).hexdigest()
    m_name = f"{m_hash}"
    return m_name


def main():
    args = parse_args()
    train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="LunarLander-v2")
    parser.add_argument("-nm", "--num_models", type=int, default=10)
    parser.add_argument("-m", "--model_name", type=str, default="ppo")
    parser.add_argument("-ne", "--num_epochs", type=int, default=150)
    parser.add_argument("--val_check_interval", type=int, default=25)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--wandb_project", type=str, default="rl_project")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    return parser.parse_args()


if __name__ == "__main__":
    main()
